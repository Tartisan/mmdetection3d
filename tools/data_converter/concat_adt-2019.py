import pickle

db_infos01 = pickle.load(open('data/adt-2019-all/kitti_format/01/adt-2019_dbinfos_train.pkl', 'rb'))
db_infos02 = pickle.load(open('data/adt-2019-all/kitti_format/02/adt-2019_dbinfos_train.pkl', 'rb'))

all_db_infos = dict()
for name, infos in db_infos01.items():
    if name not in all_db_infos:
        all_db_infos[name] = []
    for info in infos:
        info['path'] = info['path'].replace('adt-2019_gt_database', 'adt-2019_gt_database_01')
        all_db_infos[name].append(info)

for name, infos in db_infos02.items():
    if name not in all_db_infos:
        all_db_infos[name] = []
    for info in infos:
        info['path'] = info['path'].replace('adt-2019_gt_database', 'adt-2019_gt_database_02')
        all_db_infos[name].append(info)

for k, v in all_db_infos.items():
    print(f'load {len(v)} {k} database infos')

with open('data/adt-2019-all/kitti_format/adt-2019_dbinfos_train.pkl', 'wb') as f:
    pickle.dump(all_db_infos, f)