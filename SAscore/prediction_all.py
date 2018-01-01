import pickle
import theano
import numpy
from theano.gof import graph
import theano.sandbox.cuda
from dbn import DBN
from scipy.stats.stats import pearsonr
import scipy.io as sio

device = 'gpu0'
theano.sandbox.cuda.use(device)

with open('train-set.pkl', 'rb') as in_strm:
    test = pickle.load(in_strm)
X = test[0].eval()
Y = test[1].eval()

def save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol = pickle.HIGHEST_PROTOCOL)
    print('save variable to path %s' % path)
    return None

def predict(X_test, filename='best_model_reg_2.pkl'):
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

predicted_values = predict(X,'best_model_reg_3.pkl')
true_values = Y

predicted_values = numpy.asarray(predicted_values*10)
true_values = numpy.asarray(true_values*10)

result_dict = {'true': true_values, 'pred': predicted_values}

sio.savemat('result_train.mat', {'result_train':result_dict})

#save(true_values,'true-test-values.pkl')
#save(predicted_values,'predicted-test-values.pkl')

#cov = numpy.cov(true_values,predicted_values)
#varx = numpy.var(true_values)
#vary = numpy.var(predicted_values)
#pearson = cov/(varx*vary)

pearson = pearsonr(true_values,predicted_values)[0]

print('Predicted value: ' + str(predicted_values) +
        ' True value: ' + str(true_values) +
        ' Persons correlation coefficient: ' + str(pearson) +
        ' Mean Absolute error: ' + str(numpy.mean(numpy.absolute(predicted_values-true_values))))


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

def draw(freq_dict,data,k):
    my_x = []
    my_y = []
    temp_key = keys[k]
    temp_list = freq_dict[temp_key]
    l = len(temp_list)
    for i in range(l):
        temp_idx = temp_list[i]
        my_x.append(data[0][temp_idx])
        my_y.append(data[1][temp_idx])
    my_x = numpy.asarray(my_x)
    my_y = numpy.asarray(my_y)
    draw_data = (my_x,my_y)
    return draw_data

def int_prediction(freq_dict,data,interval,filename = 'best_model_reg_2.pkl'):
    int_data = draw(freq_dict, data, interval)
    int_pred = predict(int_data[0], filename)
    int_true = int_data[1]
    return int_pred,int_true

int_stats = numpy.zeros((9,2)) # intervaled mean absolute error and persons correlation coeffcient
for k in range(9):
    p, t = int_prediction(freq_dict,data,k)
    p = numpy.asarray(p*10)
    t = numpy.asarray(t*10)
    mae = numpy.mean(numpy.absolute(p-t))
    pcc = pearsonr(t,p)[0]
    int_stats[k,0] = mae
    int_stats[k,1] = pcc
    print('Mean Absolute error for interval ' + str(keys[k]) + ': \n' + str(mae) +
          '\n'
          'Persons correlation coefficient for interval ' + str(keys[k]) + ': \n' + str(pcc))
