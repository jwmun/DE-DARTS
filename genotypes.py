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

# D-DARTS(2nd order) cell
D_DARTS = Genotype(
  normal=[
    ('sep_conv_3x3', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 0),
    ('sep_conv_3x3', 1),
    ('skip_connect', 0),
    ('sep_conv_3x3', 1),
    ('sep_conv_3x3', 1),
    ('sep_conv_3x3', 0)
  ],
  normal_concat=range(2, 6),
  reduce=[
    ('max_pool_3x3', 0),
    ('sep_conv_5x5', 1),
    ('max_pool_3x3', 0),
    ('dil_conv_5x5', 2),
    ('dil_conv_5x5', 3),
    ('skip_connect', 2),
    ('skip_connect', 2),
    ('sep_conv_3x3', 3)
  ],
  reduce_concat=range(2, 6))

# D-DARTS(1st order) cell
D_DARTS_1st = Genotype(
  normal=[
    ('sep_conv_3x3', 1),
    ('dil_conv_3x3', 0),
    ('sep_conv_3x3', 1),
    ('sep_conv_5x5', 0),
    ('sep_conv_5x5', 0),
    ('sep_conv_5x5', 1),
    ('sep_conv_3x3', 1),
    ('sep_conv_3x3', 0)
  ],
  normal_concat=range(2, 6),
  reduce=[
    ('max_pool_3x3', 0),
    ('sep_conv_5x5', 1),
    ('max_pool_3x3', 0),
    ('dil_conv_5x5', 2),
    ('max_pool_3x3', 0),
    ('sep_conv_5x5', 2),
    ('max_pool_3x3', 0),
    ('skip_connect', 2)
  ],
  reduce_concat=range(2, 6))

# noise(2nd order) cell
noise = Genotype(
  normal=[
    ('sep_conv_3x3', 1),
    ('sep_conv_5x5', 0),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_3x3', 1),
    ('max_pool_3x3', 0),
    ('sep_conv_3x3', 1)
  ],
  normal_concat=range(2, 6),
  reduce=[
    ('sep_conv_5x5', 0),
    ('sep_conv_5x5', 1),
    ('max_pool_3x3', 0),
    ('sep_conv_5x5', 2),
    ('sep_conv_5x5', 2),
    ('max_pool_3x3', 0),
    ('max_pool_3x3', 0),
    ('sep_conv_3x3', 2)
  ],
  reduce_concat=range(2, 6))

