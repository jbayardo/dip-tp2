import os
for p in range(20):
    print p
    os.system('../../../src/jj_progressive.py -d out.j out%.2u.png -p %u' % (p, p))

