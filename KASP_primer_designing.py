import pandas as pd
import re

data=pd.read_csv('KASP_primer_design_homework.csv')
iupac={'R':['A','G'],'Y':['C','T'],'S':['G','C'],'W':['A','T'],'K':['G','T'],'M':['A','C']}

p1,p2,s1,s2,amplicon_one,tm1,tm2=[],[],[],[],[],[],[]

def primer(primer_one,primer_two,s1,s2,tm1,tm2,amplicon_one):
	data['Allele 1 Primer'],data['Allele 2 Primer'],data['Allele 1'],data['Allele 2 '],data['tm1'],data['tm2'],data['amplicon']=primer_one,primer_two,s1,s2,tm1,tm2,amplicon_one
def tmp_cal(msr_tmp):return (2*(msr_tmp.count('A')+msr_tmp.count('T')))+(4*(msr_tmp.count('G')+msr_tmp.count('C')))
def short(msr_tmp): return msr_tmp[1:]

def add_to_lst(d_temp,t,passed_primer,lst):
	if t < (d_temp+2) and t > (d_temp-2) :
		lst.append(passed_primer)
	else:
		passed_primer=short(passed_primer)
		t=tmp_cal(passed_primer)
		if t <= d_temp:lst.append(passed_primer)
		else:add_to_lst(d_temp,t,passed_primer,lst)	

def append_var(snp1,snp2,primer_one,primer_two,a_seq):
	s1.append(snp1)
	s2.append(snp2)
	p1.append(primer_one)
	p2.append(primer_two)
	amplicon_one.append(a_seq)

def type_one(seq,loc,cnt):
	snp1,snp2=(seq[loc.span()[0]+1],seq[loc.span()[0]+3])
	primer_one, primer_two=seq[loc.span()[0]-24:loc.span()[0]-1]+snp1, seq[loc.span()[0]-24:loc.span()[0]-1]+snp2
	d_temp=data['Desired Tm for all three primers (+-2 degrees OK)'].loc[cnt]
	
	dampl=(data['Desired amplicon length'].loc[cnt])+1
	try:a_start,a_end=loc.span()[0]+5,loc.span()[0]+(dampl+4)
	except Exception:a_start,a_end=loc.span()[0]-dampl,loc.span()[0]-4
	a_seq=seq[a_start:a_end]
	t1= tmp_cal(primer_one)
	add_to_lst(d_temp,t1,primer_one,tm1)
	t2=tmp_cal(primer_two)
	add_to_lst(d_temp,t2,primer_two,tm2)
	append_var(snp1,snp2,primer_one,primer_two,a_seq) 

def type_two(i,tmp,cnt):
	snp1,snp2=tmp[0],tmp[1]
	primer_one,primer_two=seq[i-24:i-1:],seq[i-24:i-1:]
	d_temp=data['Desired Tm for all three primers (+-2 degrees OK)'].loc[cnt]
	
	dampl=(data['Desired amplicon length'].loc[cnt])+1
	a_seq=seq[i+1:i+dampl]
	t1= tmp_cal(primer_one)
	add_to_lst(d_temp,t1,primer_one,tm1)
	t2=tmp_cal(primer_two)
	add_to_lst(d_temp,t2,primer_two,tm2)
	append_var(snp1,snp2,primer_one,primer_two,a_seq)

cnt=0
for seq in data['SourceSeq']:
	loc=re.search(r'\[',seq)
	if loc is not None:
		type_one(seq,loc,cnt)
	else:
		se=re.sub(r'[ATGCN]','',seq)
		i=seq.index(se)
		tmp=iupac[se]
		type_two(i,tmp,cnt)
	cnt+=1	

primer(p1,p2,s1,s2,tm1,tm2,amplicon_one)

pc=[]
for x in range(len(data['amplicon'])):
	
	data['amplicon'].loc[x]=data['tm1'].loc[x]+data['amplicon'].loc[x]
	red_primer_l=len(data['amplicon'].loc[x])-len(data['tm1'].loc[x])
	data['amplicon'].loc[x]= data['amplicon'].loc[x][:red_primer_l:]
	pc.append(data['amplicon'].loc[x][len(data['amplicon'].loc[x])-len(data['tm1'])::])
data['Common Primer']=pc

data.to_csv('rnadom.csv')
