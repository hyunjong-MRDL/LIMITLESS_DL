from utils import extract_data, extract_beam_moments

'''Merges data from 9 fields into ONE sigle sequence (for specific scan number)'''
'''Returns (Times, Amps, Beamtimes, States)'''
def merge_data(dict, scan_num):
    field_list = dict[scan_num]
    merged_time, merged_amp = [], []
    merged_beamtime, merged_state = [], []
    for field in field_list:
        period_of_prev_field = 0.0 if ( len(merged_time) == 0 ) else ( merged_time[-1] + 0.015 )
        time, amplitude = extract_data(field)
        beamtime, state = extract_beam_moments(field)
        merged_time += [x + period_of_prev_field for x in time]
        merged_amp += amplitude
        merged_beamtime += [x + period_of_prev_field for x in beamtime]
        merged_state += state

    return merged_time, merged_amp, merged_beamtime, merged_state

'''Takes Merged data as input argument'''
'''Returns effective (Times, Beams) when the beam is enabled'''
def split_beams(Time, Amp, Beams, States):
    beam_ID = 1
    time_dict, beam_dict = {}, {}
    for state_index in range(len(States)-1):
        curr_state = States[state_index]
        curr_beamtime, next_beamtime = Beams[state_index], Beams[state_index+1]
        if curr_state == 1:
            curr_timelist, curr_beamlist = [], []
            for amp_index in range(len(Amp)):
                if (Time[amp_index] >= curr_beamtime) and (Time[amp_index] <= next_beamtime):
                    curr_timelist.append(Time[amp_index])
                    curr_beamlist.append(Amp[amp_index])
        elif curr_state == 0:
            time_dict[beam_ID] = curr_timelist
            beam_dict[beam_ID] = curr_beamlist
            beam_ID += 1
    return time_dict, beam_dict