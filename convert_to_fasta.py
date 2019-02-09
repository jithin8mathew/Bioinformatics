import textwrap

raw_file=open('Input_File.txt').read()

raw_file=raw_file.split('>')
for items in raw_file:
	items=items.split('\n')
	seq=""
	a=(items[0])
	del items[0]
	for x in items:seq+=x
	with open('Output_folder//file.fasta','a+') as fas:
		fas.write(">"+a+"\n")
		fas.write('\n'.join(textwrap.wrap(seq, 60, break_long_words=True)))
		fas.write('\n\n')	

