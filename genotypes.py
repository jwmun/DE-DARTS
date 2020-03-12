from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

NASNet = Genotype(
  normal = [
    ('sep_conv_5x5', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 0),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 0),
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    ('sep_conv_5x5', 1),
    ('sep_conv_7x7', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('sep_conv_5x5', 0),
    ('skip_connect', 3),
    ('avg_pool_3x3', 2),
    ('sep_conv_3x3', 2),
    ('max_pool_3x3', 1),
  ],
  reduce_concat = [4, 5, 6],
)
    
AmoebaNet = Genotype(
  normal = [
    ('avg_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 2),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 3),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 1),
    ],
  normal_concat = [4, 5, 6],
  reduce = [
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('max_pool_3x3', 0),
    ('sep_conv_7x7', 2),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('max_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('conv_7x1_1x7', 0),
    ('sep_conv_3x3', 5),
  ],
  reduce_concat = [3, 4, 6]
)

DARTS_V1 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])

my_darts = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 2), ('sep_conv_3x3', 3), ('max_pool_3x3', 0), ('sep_conv_5x5', 2), ('max_pool_3x3', 0)], reduce_concat=range(2, 6))

my_darts2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('skip_connect', 0), ('sep_conv_5x5', 2), ('dil_conv_5x5', 2), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))
darts_decay_softmax = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('sep_conv_3x3', 2), ('max_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))
my_darts_Dynamic = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('max_pool_3x3', 0), ('sep_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))
my_Dynamic_alpha_normal = ([[0.0841, 0.1646, 0.0465, 0.1068, 0.2317, 0.1367, 0.1050, 0.1247],
        [0.2715, 0.0568, 0.0360, 0.0639, 0.2468, 0.1280, 0.0969, 0.1002],
        [0.1129, 0.1470, 0.0608, 0.1318, 0.2004, 0.1462, 0.1058, 0.0951],
        [0.3406, 0.0742, 0.0401, 0.0685, 0.1148, 0.1214, 0.1178, 0.1225],
        [0.3319, 0.0716, 0.0337, 0.0872, 0.1272, 0.1288, 0.1049, 0.1148],
        [0.1702, 0.1428, 0.0527, 0.1060, 0.1713, 0.1597, 0.1073, 0.0900],
        [0.3852, 0.0804, 0.0448, 0.0757, 0.1108, 0.1130, 0.1128, 0.0773],
        [0.4421, 0.0592, 0.0307, 0.0734, 0.1477, 0.0789, 0.0825, 0.0855],
        [0.5233, 0.0372, 0.0281, 0.0485, 0.0971, 0.0974, 0.0805, 0.0878],
        [0.3176, 0.1352, 0.0466, 0.0875, 0.1231, 0.0991, 0.0929, 0.0981],
        [0.5141, 0.0654, 0.0369, 0.0641, 0.0690, 0.0875, 0.0843, 0.0787],
        [0.5655, 0.0559, 0.0230, 0.0569, 0.0909, 0.0700, 0.0723, 0.0654],
        [0.6753, 0.0295, 0.0201, 0.0321, 0.0626, 0.0520, 0.0569, 0.0714],
        [0.7298, 0.0194, 0.0174, 0.0216, 0.0570, 0.0556, 0.0453, 0.0539]],
       )
my_Dynamic_alpha_reduce = ([[0.0749, 0.2475, 0.1176, 0.0935, 0.1138, 0.1359, 0.1143, 0.1024],
        [0.1656, 0.1218, 0.0759, 0.1314, 0.1271, 0.1419, 0.1109, 0.1254],
        [0.0797, 0.2113, 0.1339, 0.1255, 0.1137, 0.1242, 0.1120, 0.0997],
        [0.1289, 0.1128, 0.0905, 0.1145, 0.1679, 0.1460, 0.0981, 0.1413],
        [0.1441, 0.0969, 0.0661, 0.1176, 0.1198, 0.1519, 0.1489, 0.1547],
        [0.0771, 0.1640, 0.1468, 0.1066, 0.1407, 0.1506, 0.1039, 0.1103],
        [0.1357, 0.1187, 0.1112, 0.1189, 0.1140, 0.1411, 0.1107, 0.1497],
        [0.1357, 0.0973, 0.0841, 0.1390, 0.1310, 0.1608, 0.1096, 0.1423],
        [0.1984, 0.0797, 0.0737, 0.1203, 0.1426, 0.1335, 0.1117, 0.1401],
        [0.0878, 0.2004, 0.1720, 0.1032, 0.1307, 0.0932, 0.1120, 0.1007],
        [0.1263, 0.1341, 0.1152, 0.1642, 0.1171, 0.1223, 0.1100, 0.1109],
        [0.1377, 0.0895, 0.0722, 0.1388, 0.1274, 0.1233, 0.1111, 0.2001],
        [0.1863, 0.0765, 0.0687, 0.1310, 0.1286, 0.1367, 0.1230, 0.1492],
        [0.2038, 0.0646, 0.0653, 0.1075, 0.1055, 0.1510, 0.1352, 0.1671]],
)
darts_updecay_sigmoid = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))
darts_test1 = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('skip_connect', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))
darts_test2 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 2), ('dil_conv_5x5', 3), ('skip_connect', 2), ('skip_connect', 2), ('sep_conv_3x3', 3)], reduce_concat=range(2, 6))
darts_f_order = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 3), ('sep_conv_5x5', 0), ('max_pool_3x3', 0), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))
softmax_dynamic_1st_darts_cell_correct = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 2), ('dil_conv_3x3', 3), ('skip_connect', 2), ('sep_conv_3x3', 0)], reduce_concat=range(2, 6))
darts_noise1 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('sep_conv_3x3', 2)], reduce_concat=range(2, 6))
darts_noise2 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_5x5', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0)], reduce_concat=range(2, 6))
darts_1st3 = Genotype(normal=[('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('sep_conv_5x5', 2), ('max_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))

DARTS = darts_test2

