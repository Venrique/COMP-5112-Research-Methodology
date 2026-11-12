"""Full-split, resumable NonLocalMeans validation with fixed batched decoding."""
import hashlib
import json
from pathlib import Path
from time import perf_counter

import pandas as pd
from PIL import Image

from src.evaluation.recognition import corpus_metrics
from src.preprocessing import NonLocalMeans


def configurations() -> dict:
    """Five planned contrasts; each window/color contrast uses the h=3 anchor."""
    result = {f'nlm_h{h}': NonLocalMeans(h=float(h)) for h in (1, 3, 7)}
    result.update({
        'nlm_template3': NonLocalMeans(template_window_size=3),
        'nlm_search11': NonLocalMeans(search_window_size=11),
    })
    return {'baseline': None, **result}


def adapt_image(image: Image.Image) -> Image.Image:
    """White composite for RGBA; preserve L/RGB and never modify source pixels."""
    if image.mode == 'RGBA':
        return Image.alpha_composite(Image.new('RGBA', image.size, 'white'), image).convert('RGB')
    if image.mode not in ('L', 'RGB'):
        raise ValueError(f'Unexpected mode: {image.mode}')
    return image.copy()


def predict_batch(model, images: list) -> list[str]:
    """Same processor and generation options as TrOCRRecognizer.predict."""
    import torch
    pixels = model.processor(images=[im.convert('RGB') for im in images], return_tensors='pt').pixel_values
    with torch.inference_mode():
        ids = model.model.generate(pixels.to(model.device), **model.generation_kwargs)
    return model.processor.batch_decode(ids, skip_special_tokens=True)


def evaluate(dataset, dataset_name: str, model, configuration: str, transform,
             output_dir: Path, batch_size: int = 8) -> dict:
    """Save each batch; resume only after checking the complete run identity."""
    folder = output_dir / dataset_name / configuration
    folder.mkdir(parents=True, exist_ok=True)
    identity = {'dataset_fingerprint': dataset._fingerprint, 'count': len(dataset),
                'model': model.to_config(), 'transform': transform.to_config() if transform else None,
                'batch_size': batch_size, 'normalization': 'collapse whitespace; preserve case/punctuation',
                'input_adaptation': 'RGBA white composite; preserve L/RGB'}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    manifest = folder / 'identity.json'
    if manifest.exists() and json.loads(manifest.read_text())['signature'] != signature:
        raise ValueError(f'Resume identity mismatch: {folder}')
    manifest.write_text(json.dumps({'signature': signature, **identity}, indent=2))
    path = folder / 'predictions.csv'
    rows = pd.read_csv(path, keep_default_na=False).to_dict('records') if path.exists() else []
    if [int(r['index']) for r in rows] != list(range(len(rows))) or len(rows) > len(dataset):
        raise ValueError('Invalid prediction checkpoint order')
    for start in range(len(rows), len(dataset), batch_size):
        images, current = [], []
        for index in range(start, min(start + batch_size, len(dataset))):
            sample = dataset[index]
            if not isinstance(sample['text'], str) or not sample['text'].strip():
                raise ValueError(f'Missing ground truth at {dataset_name}/{index}')
            begin = perf_counter()
            image = adapt_image(sample['image'])
            if transform is not None:
                image = transform(image)
            images.append(image)
            current.append({'index': index, 'sample_id': sample.get('id', f'sample_{index}'),
                            'ground_truth': sample['text'], 'preprocessing_seconds': perf_counter() - begin})
        begin = perf_counter()
        predictions = predict_batch(model, images)
        seconds = perf_counter() - begin
        if len(predictions) != len(current):
            raise ValueError('Prediction count mismatch')
        for row, prediction in zip(current, predictions):
            rows.append({**row, 'prediction': prediction, 'inference_seconds': seconds / len(current),
                         **corpus_metrics([row['ground_truth']], [prediction])})
        temporary = path.with_suffix('.tmp')
        pd.DataFrame(rows).to_csv(temporary, index=False)
        temporary.replace(path)
        if start == 0 or len(rows) % (batch_size * 10) == 0 or len(rows) == len(dataset):
            print(f'{dataset_name}/{configuration}: {len(rows)}/{len(dataset)}', flush=True)
    summary = {'dataset': dataset_name, 'configuration': configuration, 'sample_count': len(rows),
               **corpus_metrics([r['ground_truth'] for r in rows], [r['prediction'] for r in rows]),
               'preprocessing_seconds': sum(r['preprocessing_seconds'] for r in rows),
               'inference_seconds': sum(r['inference_seconds'] for r in rows)}
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2))
    return summary


def summarize(summaries: list[dict], output_dir: Path) -> tuple:
    """Equal-dataset ranking avoids weighting the larger corpus more heavily."""
    table = pd.DataFrame(summaries)
    for metric in ('cer', 'wer'):
        baseline = table[table.configuration == 'baseline'].set_index('dataset')[metric]
        table[f'{metric}_delta_vs_baseline'] = table[metric] - table.dataset.map(baseline)
    table.to_csv(output_dir / 'comparison.csv', index=False)
    ranking = table.groupby('configuration')[['cer', 'wer']].mean().sort_values(['cer', 'wer'])
    ranking.to_csv(output_dir / 'equal_dataset_ranking.csv')
    winners = table.sort_values(['cer', 'wer']).groupby('dataset').first().reset_index()
    winners.to_csv(output_dir / 'dataset_winners.csv', index=False)
    return table, ranking, winners
