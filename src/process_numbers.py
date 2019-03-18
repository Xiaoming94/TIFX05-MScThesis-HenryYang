import utils

dict_test = { 'Hello' : 'World' }

utils.save_processed_data(dict_test,"test_save")
dict_test2 = utils.load_processed_data("test_save")

print(dict_test2)
print(dict_test)