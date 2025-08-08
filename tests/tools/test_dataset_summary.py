from tools.dataset_summary import dataset_summary

#Fake data to create for tests
def test_dataset_summary() -> str:
    result = dataset_summary('dataset/DD_EEC_ANNUEL_2024_data.csv')
    return result
    

if __name__ == "__main__":
    result = test_dataset_summary()
    print(result)