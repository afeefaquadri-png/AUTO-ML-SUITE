import pytest
import pandas as pd
from modules.data_ingestion import load_csv
import tempfile
import os

def test_load_csv():
    # Create a temp CSV
    data = {'col1': [1, 2], 'col2': ['a', 'b']}
    df = pd.DataFrame(data)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f, index=False)
        temp_path = f.name
    
    try:
        loaded_df = load_csv(temp_path)
        pd.testing.assert_frame_equal(df, loaded_df)
    finally:
        os.unlink(temp_path)