import pandas as pd
import re

data = open('comparison.txt','r+').read()

data = re.findall(r'(\d{1,}\s[\+|\-]\s{1,}\d\s[A-z]{1,}\s{1,}\d{1,}\s\-\s{1,}\d{1,}\s{1,}\d{1,}\.\d{1,}\s{1,}\d{1,}\s{1,}\-\s{1,}\d{1,}\s{1,}\d{1,})',data)
data2=[]
for x in data:
	tmp_lst=[]
	[tmp_lst.append(y) for y in x.split(' ') if y != '']
	data2.append(tmp_lst)

data = pd.DataFrame(data2, index= [x for x in range(len(data))], columns=['G','Str','exon_no','Feature','Start','hyphen','End','Score','ORF_start','hyphen2','ORF_end','Len'])
data=data.drop(columns=['hyphen','hyphen2'])

# Uncomment the below line to generate csv file of exon sequence data
#data.to_csv('Arabidopsis_exons.csv')

data= data.groupby(['G'])
print(data)

for key, item in data:
    print(data.get_group(key), "\n\n")