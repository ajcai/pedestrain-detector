from __future__ import print_function
from __future__ import absolute_import
import ConfigParser as cp
import os




def batch(det_name):
    #sample image patches
    #os.system('python annotationutil.py')
    # #extract features
    os.system('python feature_extractor.py')
    # #train
    os.system('python trainer.py')
    # #retrain
    os.system('python retrainer.py')
    # #save detections
    os.system('python tester.py -p {}'.format(det_name))



# orient=[3,4,6,9]
# for ort in orient:
#     conf = cp.RawConfigParser()
#     conf.read('../data/config/origin_config.cfg')
#     #modify configuration file
#     conf.set('hog','orientations',ort)
#     conf.write(open('../data/config/config.cfg','w'))

#     batch('../data/detections/detections_orient_{}.data'.format(ort))
# print('orient job finished!')

cell_sz=[[4,4],[6,6],[8,8],[10,10]]
for sz in cell_sz:
    conf = cp.RawConfigParser()
    conf.read('../data/config/origin_config.cfg')
    #modify configuration file
    conf.set('hog','pixels_per_cell',str(sz))
    conf.write(open('../data/config/config.cfg','w'))

    batch('../data/detections/detections_cell_{}.data'.format(sz))
print('cell job finished!')

block_sz=[[4,4],[3,3],[2,2]]
for sz in block_sz:
    conf = cp.RawConfigParser()
    conf.read('../data/config/origin_config.cfg')
    #modify configuration file
    conf.set('hog','cells_per_block',str(sz))
    conf.write(open('../data/config/config.cfg','w'))

    batch('../data/detections/detections_block_{}.data'.format(sz))
print('cell block finished!')

# norm_type=['L1','L1-sqrt','L2','L2-Hys']
# for n in norm_type:
#     conf = cp.RawConfigParser()
#     conf.read('../data/config/origin_config.cfg')
#     #modify configuration file
#     conf.set('hog','normalize',n)
#     conf.write(open('../data/config/config.cfg','w'))

#     batch('../data/detections/detections_norm_{}.data'.format(n))
# print('norm job finished!')