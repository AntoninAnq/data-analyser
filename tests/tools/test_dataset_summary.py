from tools.dataset_summary import dataset_summary

#Fake data to create for tests
def test_dataset_summary():
    """Test that dataset_summary function returns a string result"""
    result = dataset_summary('dataset/DD_EEC_ANNUEL_2024_data.csv')
    assert isinstance(result, str)
    assert len(result) > 0
    assert "dataset" in result.lower() or "column" in result.lower() or "row" in result.lower()
    

if __name__ == "__main__":
    result = dataset_summary('dataset/DD_EEC_ANNUEL_2024_data.csv')
    print(result)