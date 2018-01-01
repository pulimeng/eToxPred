import pickle
import theano
import numpy
import random
from theano.gof import graph
from dbn import DBN

with open('sa-data-part.pkl', 'rb') as in_strm:
    data = pickle.load(in_strm)
sa = data[1]
keys = ['1-2','2-3','3-4','4-5','5-6','6-7','7-8','8-9','9-10']
def hist_counts(sa,keys):
    freq_dict = dict.fromkeys(keys,[])
    for k in range(len(sa)):
        item = sa[k]
        item = item * 10
        idx = round(item)
        if idx - item >0:
            real_key = keys[int(idx-1-1)]
        if idx - item <0:
            real_key = keys[int(idx-1)]
        freq_dict[real_key]=freq_dict[real_key]+[k]
    return freq_dict

freq_dict = hist_counts(sa,keys)

def rand_draw(freq_dict,data):
    my_x = []
    my_y = []
    for k in range(9):
        temp_key = keys[k]
        temp_list = freq_dict[temp_key]
        l = len(temp_list)
        idx = random.randint(0,l)
        v = temp_list[idx]
        my_x.append(data[0][v])
        my_y.append(data[1][v])
    my_x = numpy.asarray(my_x)
    my_y = numpy.asarray(my_y)
    draw = (my_x,my_y)
    return draw

def predict(X_test, filename='best_model_reg_1.pkl'):
    # load the saved model
    model_file = open(filename, 'rb')
    with open(filename, 'rb') as in_strm:
        regressor = pickle.load(in_strm)
    model_file.close()
    y_pred = regressor.linearLayer.y_pred

    # find the input to theano graph
    inputs = graph.inputs([y_pred])
    # select only x
    inputs = [item for item in inputs if item.name == 'x']
    # compile a predictor function
    predict_model = theano.function(
        inputs=inputs,
        outputs=y_pred)
    X_test = X_test.astype(numpy.float32)
    predicted_values = predict_model(X_test)
    return predicted_values

draw = rand_draw(freq_dict,data)
predicted_values = predict(draw[0],'best_model_3.pkl')
true_values = draw[1]

predicted_values = numpy.asarray(predicted_values*10)
true_values = numpy.asarray(true_values*10)
print('Predicted value: ' + str(predicted_values) +
        ' True value: ' + str(true_values) +
        ' Absolute error: ' + str(numpy.absolute(predicted_values-true_values)))