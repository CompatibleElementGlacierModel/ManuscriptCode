import numpy as np
import matplotlib.pyplot as plt


p1_z = np.linspace(0,240,25)
p1_d = np.array([315,279,369,394,352,350,369,357,
    383,365,384,380,393,431,417,413,406,422,435,
    430,454,469,469,530])

p1h1_z = np.array([260,280,300,320,343,360,380,402,421,
    440,454,476,498,520,534,553,574,597,617,636,665,681,
    695,723,741])
p1h1_l = p1h1_z[1:] - p1h1_z[:-1]

p1h1_w = np.array([400,395,405,465,275,362,455,413,394,
    272,498,460,460,275,403,467,529,513,435,652,366,353,658,455])

p1h2_z = np.array([260,283,303,320,340,352,364,384,411,438,453,474,487,499,519,537,552,572,584])
p1h2_w = np.array([455,394,348,411,227,232,435,540,556,359,460,304,280,446,383,388,383,260])
p1h2_l = np.array([23,20,17,20,12,12,20,27,27,15,21,13,12,20,18,15,20,14])


p2_z = np.linspace(0,160,17)
p2_d = np.array([422,287,342,369,388,384,396,364,377,412,401,413,419,414,447,433])

p2h1_z = np.array([170,185,213,233,255,280,294,320,340,357,380,406,432,460,480,501,521,541,560,592,621])
p2h1_l = p2h1_z[1:] - p2h1_z[:-1]
p2h1_w = np.array([267,500,405,438,472,289,503,374,320,401,439,551,569,407,403,391,558,426,671,678])

p3_z = np.linspace(0,170,18)
p3_d = np.array([350,329,362,362,342,367,361,396,387,395,412,436,442,411,441,469,436])
p3h1_z = np.array([190,221,251,273,289,306,326,346,361,381,405,425,449,459,479,505,528,548,566,591,609,637,648,667,681,695])
p3h1_l = np.array([21,30,23,16,17,20,20,15,20,24,20,19,10,20,26,23,20,18,25,18,28,11,19,14,14])
p3h1_w = np.array([439,552,459,343,346,410,422,308,449,461,415,372,207,412,532,487,404,404,546,459,706,261,470,349,335])

kovacs_diameter = 7.5
kovacs_area = (kovacs_diameter/2.)**2*np.pi

p1h1_v = p1h1_l*kovacs_area
p1h2_v = p1h2_l*kovacs_area
p2h1_v = p2h1_l*kovacs_area
p3h1_v = p3h1_l*kovacs_area

p1h1_d = p1h1_w/p1h1_v*1000
p1h2_d = p1h2_w/p1h2_v*1000
p2h1_d = p2h1_w/p2h1_v*1000
p3h1_d = p3h1_w/p3h1_v*1000


p1_xy = np.array([60.381139,-140.277101,1651])
p2_xy = np.array([60.387577,-140.292982,1653])
p3_xy = np.array([60.375174,-140.245856,1641])
p1_date = '2023-05-14'
p2_date = '2023-05-16'
p3_data = '2023-05-17'


p1_dict = {'coordinates':p1_xy,
           'pit_wall':{ 'bounds': p1_z,
                        'density': p1_d},
           'core_1':{'bounds':p1h1_z,
                     'density': p1h1_d},
           'core_2':{'bounds':p1h2_z,
                     'density': p1h2_d}}

p2_dict = {'coordinates':p2_xy,
           'pit_wall':{ 'bounds': p2_z,
                        'density': p2_d},
           'core_1':{'bounds': p2h1_z,
                     'density': p2h1_d}
           }

p3_dict = {'coordinates':p3_xy,
           'pit_wall':{ 'bounds': p3_z,
                        'density': p3_d},
           'core_1':{'bounds': p3h1_z,
                     'density': p3h1_d}
           }



plt.stairs(p1_d,p1_z,orientation='horizontal',color='red',baseline=None,label='Pit 1, Core 1')
plt.stairs(p1h1_d,p1h1_z,orientation='horizontal',color='red',baseline=None)
plt.stairs(p1h2_d,p1h2_z,orientation='horizontal',color='green',baseline=None,label='Pit 1, Core 2')
plt.stairs(p2_d,p2_z,orientation='horizontal',color='blue',baseline=None,label='Pit 2')
plt.stairs(p2h1_d,p2h1_z,orientation='horizontal',color='blue',baseline=None)
plt.stairs(p3_d,p3_z,orientation='horizontal',color='orange',baseline=None,label='Pit 3')
plt.stairs(p3h1_d,p3h1_z,orientation='horizontal',color='orange',baseline=None)
plt.axhline(454,color='k',linestyle='--',label='2022 Surface')
plt.axhline(740,color='k',linestyle=':',label='2021 Surface')
#plt.barh(p1h1_z[:-1],p1h1_d,height=p1h1_l,align='edge',color='none',edgecolor='r')
#plt.barh(p2_z[:-1],p2_d,height=10,align='edge',color='none',edgecolor='b')
#plt.barh(p2h1_z[:-1],p2h1_d,height=p2h1_l,align='edge',color='none',edgecolor='b')
plt.gca().invert_yaxis()
plt.xlabel('$\\rho$ (kg/m$^3$)')
plt.ylabel('Depth (cm)')
plt.legend()

fig = plt.gcf()
fig.set_size_inches(6,6)
fig.savefig('malaspina_pits.png',dpi=300)
plt.show()

