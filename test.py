from features.pulse import Hemodynamics

hemo = Hemodynamics()
res = hemo.do('test_data/2013711001.txt')
print(res)