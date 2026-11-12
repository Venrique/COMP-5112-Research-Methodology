"""Freeze an active treatment using validation-only, dataset-specific scores."""
import pandas as pd


def select_validation_configuration(table: pd.DataFrame, dataset: str) -> dict:
    """Choose lowest CER, then WER; exact ties retain the original grid order.

    Baseline stays a separate experimental control. It is excluded here so a
    technique is still measured even when no active setting beat validation
    baseline. This function must only receive validation-study summaries.
    """
    rows = table[(table['dataset'] == dataset) & (table['configuration'] != 'baseline')]
    if rows.empty or rows[['cer', 'wer']].isna().any().any():
        raise ValueError(f'Missing active validation scores for {dataset}')
    if (rows[['cer', 'wer']] < 0).any().any():
        raise ValueError('Error rates must be nonnegative')
    return rows.sort_values(['cer', 'wer'], kind='stable').iloc[0].to_dict()
