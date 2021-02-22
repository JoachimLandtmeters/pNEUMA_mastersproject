import pandas as pd


def cleaning_counting(det_c,  n_det , double_loops=False):
    det_cross = []
    indexes_list=[]
    crossed = {}
    for det in range(1, n_det + 1):
        for a, b in det_c.iterrows(): # For every time step
            if double_loops:
                t_cross = [[b[f'times_{det}'][e], b[f'times_lp_{det}'][e]] for e in
                           range(0, len(b['times_1']))] # Do for every trajectory
                det_cross.append(t_cross)
                #d_travel = [[b[f'counts_{det}'][e], b[f'counts_lp_{det}'][e]] for e in
                            #range(0, len(b['counts_1']))]
        crossed[f'no_cross_{det}'] = []
        crossed[f'partly_cross_{det}'] = []
        for ind in range(0, len(det_cross[0])): # For every trajectory
            cr=False
            cr_lp=False
            #if double_loops:
                #crossed['cross_' + str(det)] = []
                #crossed['cross_lp_' + str(det)] = []
                #crossed['loc_step_' + str(det)]=[]
            for i in range(len(det_c) * (det - 1), len(det_c) * det): # For every time step for specific detector pair
                if bool(det_cross[i][ind][0]):
                    cr=True
                if bool(det_cross[i][ind][1]):
                    cr_lp=True
                """ 
                if double_loops:
                    crossed['cross_' + str(det)].append([bool(det_cross[i][ind][0]),(i-len(det_c) * (det - 1))])
                    crossed['cross_lp_' + str(det)].append([bool(det_cross[i][ind][1]),(i-len(det_c) * (det - 1))])
                if double_loops:
                    if crossed['cross_lp_' + str(det)][(i-len(det_c) * (det - 1))][0] \
                            and not crossed['cross_' + str(det)][(i-len(det_c) * (det - 1))][0]:
                        print('Did not cross first detector of pair ' + str(det) + ' : ' + str(ind))
                        cnt_cross.append((ind,crossed['cross_lp_' + str(det)][(i-len(det_c) * (det - 1))][1],int(2)))
                    if crossed['cross_' + str(det)][(i-len(det_c) * (det - 1))][0] \
                            and not crossed['cross_lp_' + str(det)][(i-len(det_c) * (det - 1))][0]:
                        print('Did not cross second detector of pair ' + str(det) + ' : ' + str(ind))
                        cnt_cross.append((ind,crossed['cross_lp_' + str(det)][(i-len(det_c) * (det - 1))][1],int(1)))
                """
            if cr and cr_lp:
                continue
            elif cr != cr_lp:
                #print('Did not cross one of detector ' + str(det) + ' edges : ' + str(ind))
                crossed[f'partly_cross_{det}'].append(ind)
            else:
                #print('Did not cross detector ' + str(det) + ' : ' + str(ind))
                crossed[f'no_cross_{det}'].append(ind)
        #indexes=[]
        #unique_indexes=[]
        #for u, v in enumerate(cnt_cross):
            #for w, x in enumerate(cnt_cross):
                #if u != w:
                    #if v[0]==x[0]:
                        #indexes.append(v[0])
        #for u in cnt_cross:
            #if u[0] not in indexes:
                #unique_indexes.append(u)
        #indexes_list.append(unique_indexes)
    #indexes_final=[pd.DataFrame(indexes_list[i]) for i in range(0,len(indexes_list))]
    return crossed
