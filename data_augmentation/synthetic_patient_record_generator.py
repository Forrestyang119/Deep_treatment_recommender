import csv
import pandas as pd 
import numpy as np 
import random
import pdb

class DataAugumentation:


    def __init__(self):
        # Input files
        alignment_file = 'Files_synthetic/alignment.csv'
        consensus_file = 'Files_synthetic/consensus_sequence.csv'
        patattr_file = 'Files_synthetic/Synthetic_patientAttributes_1000.csv'


        # get consensus sequence
        self.concensus_act = []
        self.concensus_prob = []
        with open(consensus_file, 'r') as f:
            reader = csv.reader(f, delimiter = ',')
            for row in reader:
                self.concensus_act.append(row[0])
                self.concensus_prob.append(float(row[1]))
        pat_attr_list = [[] for i in range(len(self.concensus_act))]
        # act_attr_list = [[] for i in range(len(self.concensus_act))]
     
        # get alignments
        alignment_dict = {}
        with open(alignment_file, 'r') as f:
            reader = csv.reader(f, delimiter = ',')
            for row in reader:
                alignment_dict[row[0]] = row[1:]
        if 'Percentage' in alignment_dict.keys():
            concensus_percentage = alignment_dict['Percentage']
            alignment_dict.pop('Percentage', None)
        
        # get patient attributes
        patt_attr_dict = {}
        with open(patattr_file, 'r') as f:
            reader = csv.reader(f, delimiter = ',')
            self.patient_attr_header = next(reader, None)
            for row in reader:
                patt_attr_dict[row[0]] = [int(i) for i in row[1:]]
        ids = [key for key in patt_attr_dict]

        
        for p_id in alignment_dict:
            # print(p_id, ':')
            act_trace = alignment_dict[p_id]
            act_idx = 0
            # act_attrs = act_attr_dict[p_id]
            # pdb.set_trace()
            for i in range(len(act_trace)):
                if not act_trace[i] == '-':
                    pat_attr_list[i].append(patt_attr_dict[p_id])
                    act_idx += 1
        # Normalization
        self.pat_attr_val = [[] for i in range(len(self.concensus_act))]
        self.pat_attr_pro = [[] for i in range(len(self.concensus_act))]
        for i in range(len(self.concensus_act)):
            # i th activity in concensus sequence
            patient_attributes = pat_attr_list[i]
            # activity_attributes = act_attr_list[i]
            # print(i)
            if len(patient_attributes) == 0:
                continue;
            # normalize for patient attributes
            for j in range(len(patient_attributes[0])):
                # jth patient attribute
                val_list = [patient_attributes[k][j] for k in range(len(patient_attributes))]
                # get value and probs
                from collections import Counter
                cnt = Counter(val_list)
                vals = []
                probs = []
                for k,v in cnt.items():
                    vals.append(k)
                    probs.append(v/len(val_list))
                self.pat_attr_val[i].append(vals)
                self.pat_attr_pro[i].append(probs)

    def init_output_file(self):
        column = []
        # column += self.act_attr_header[2:]
        column = [self.patient_attr_header[0]] + ['Activity'] + self.patient_attr_header[1:]
        with open('Files_synthetic/rnn_input_1000.csv', 'w',newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(column)
 
    
    def generate_data_from_trace(self, trace_file):
        self.init_output_file()
        output_result = []

        # load gengerated trace from file
        generated_trace = np.genfromtxt(trace_file, delimiter=',')

        # Generate Artificial data
        for i in range(generated_trace.shape[0]):
            arti_acts = []
            pat_attrs = []
            act_attrs = []
            n_des = []
            nth_occur = []
            # traverse over concensus sequence
            for j in range(len(self.concensus_act)):
                if len(self.pat_attr_val[j]) == 0:
                    continue
                occur_num = {act:0 for act in self.concensus_act}
                # if random.random() < self.concensus_prob[j]:
                if generated_trace[i][j] == 1:
                    # activity appears
                    arti_acts.append(self.concensus_act[j])
                    occur_num[self.concensus_act[j]] += 1
                    # get patient attributes
                    attrs_vals = self.pat_attr_val[j]
                    attrs_probs = self.pat_attr_pro[j]
                    pat_attr_generated = []
                    for k in range(len(attrs_vals)):
                        # kth attribute
                        r = random.random()
                        sum_prob = 0
                        vals = attrs_vals[k]
                        probs = attrs_probs[k]
                        for l in range(len(vals)):
                            sum_prob += probs[l]
                            if sum_prob > r:
                                pat_attr_generated.append(vals[l])
                                break
                    pat_attrs.append(pat_attr_generated)
                    
            # Build result
            # randomly choose patient attribute
            # patient_attr_out = random.choice(pat_attrs)
            patient_attr_out = []
            # print(pat_attrs)
            for j in range(len(pat_attrs[0])):
                # print(j)
                cur_attr = [pat_attrs[v][j] for v in range(len(pat_attrs))]
                # from collections import Counter
                # cnt = Counter(cur_attr)
                # patient_attr_out.append(cnt.most_common(1)[0][0])
                patient_attr_out.append(random.choice(cur_attr))
            for j in range(len(arti_acts)):
                output = []
                output += [i, arti_acts[j]]
                output += patient_attr_out
                # output += [nth_occur[j], n_des[j]]
                # output += act_attrs[j]
                # pdb.set_trace()
                output_result.append(output)
                with open('Files_synthetic/rnn_input_1000.csv', 'a',newline='') as f:
                    writer = csv.writer(f, delimiter=',')
                    # writer.writerow(column)
                    writer.writerow(output)

if __name__ == "__main__":
    data_augumentor = DataAugumentation()
    data_augumentor.generate_data_from_trace('Files_synthetic/Bernoulli_generated_1000.csv')
