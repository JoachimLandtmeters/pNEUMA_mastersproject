import pandas as pd
import time
import sys
import csv
maxInt = sys.maxsize  # to read problematic csv files with various numbers of columns
decrement = True
start = time.time()


def data_parser(path_trajectory_data):
    csv.field_size_limit(sys.maxsize)
    data_file = open(path_trajectory_data, 'r')
    data_reader = csv.reader(data_file)
    data = []
    for row in data_reader:
        data.append([elem for elem in row[0].split("; ")])
    return data


def create_new_trajectory_data_frame():
    headings_names = ['Tracked Vehicle', 'Type', 'Entry Gate', 'Entry Time[ms]', 'Exit Gate', 'Exit Time[ms]', 'Traveled Dist.[m]',  'Avg.Speed[km / h]',
                      'Latitude [deg]', 'Longitude [deg]', 'Speed[km / h]', 'Tan.Accel.[ms - 2]', 'Lat.Accel.[ms - 2]', 'Time[ms]']
    new_df = pd.DataFrame(columns=headings_names)
    return new_df, headings_names


path_input_trajectory_data = "all_network_1000_1030_25.csv"        # csv file
path_to_export = "~/Desktop/Joachim/Master Logistics and Traffic/Erasmus Lausanne/" \
                 "Masters Project/athens_project/dataset_csv/"                               # folder

trajectory_data_array = data_parser(path_input_trajectory_data)
new_trajectory, column_names = create_new_trajectory_data_frame()
tracked_vehicle_id = 0

for tracked_vehicle_id in range(1, len(trajectory_data_array)):
    print('Tracked Vehicle: ', tracked_vehicle_id)
    print(trajectory_data_array[tracked_vehicle_id][1])
    new_trajectory.at[0, column_names[0]] = trajectory_data_array[tracked_vehicle_id][0]  # 0: Tracked Vehicle
    new_trajectory.at[0, column_names[1]] = trajectory_data_array[tracked_vehicle_id][1]  # 1: Type
    new_trajectory.at[0, column_names[2]] = trajectory_data_array[tracked_vehicle_id][2]  # 2: Entry Gate
    new_trajectory.at[0, column_names[3]] = trajectory_data_array[tracked_vehicle_id][3]  # 3: Entry Time [ms]
    new_trajectory.at[0, column_names[4]] = trajectory_data_array[tracked_vehicle_id][4]  # 4: Exit Gate
    new_trajectory.at[0, column_names[5]] = trajectory_data_array[tracked_vehicle_id][5]  # 5: Exit Time [ms]
    new_trajectory.at[0, column_names[6]] = trajectory_data_array[tracked_vehicle_id][6]  # 6: Traveled Dist. [m]
    new_trajectory.at[0, column_names[7]] = trajectory_data_array[tracked_vehicle_id][7]  # 7: Avg. Speed [km/h]
    for j in range(8, len(trajectory_data_array[tracked_vehicle_id]), 6):
        try:
            new_trajectory.at[divmod(j, 6)[0] - 1, column_names[8]] = trajectory_data_array[tracked_vehicle_id][j]          # 8: Latitude [deg]
            new_trajectory.at[divmod(j, 6)[0] - 1, column_names[8 + 1]] = trajectory_data_array[tracked_vehicle_id][j + 1]  # 9: Longitude [deg]
            new_trajectory.at[divmod(j, 6)[0] - 1, column_names[8 + 2]] = trajectory_data_array[tracked_vehicle_id][j + 2]  # 10: Speed [km/h]
            new_trajectory.at[divmod(j, 6)[0] - 1, column_names[8 + 3]] = trajectory_data_array[tracked_vehicle_id][j + 3]  # 11: Tan. Accel. [ms-2]
            new_trajectory.at[divmod(j, 6)[0] - 1, column_names[8 + 4]] = trajectory_data_array[tracked_vehicle_id][j + 4]  # 12: Lat. Accel. [ms-2]
            new_trajectory.at[divmod(j, 6)[0] - 1, column_names[8 + 5]] = trajectory_data_array[tracked_vehicle_id][j + 5]  # 13: Time [ms]
        except IndexError:
            continue

    new_trajectory.to_csv(path_to_export + 'trajectory' + str(tracked_vehicle_id) + '.csv', index=False)
    new_trajectory, column_names = create_new_trajectory_data_frame()

end = time.time()
print()
print()
print('Time for extracting ' + str(tracked_vehicle_id) + ' vehicles was ' + str(int(divmod(end - start, 60)[0])) + ' minutes and ' +
      str(int(divmod(end - start, 60)[1])) + ' seconds.')
print()
