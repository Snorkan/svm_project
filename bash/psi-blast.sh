#!/usr/bin/env

export BLASTDB=/local_uniref/uniref/uniref90

echo $PATH
export PATH=$PATH:directory
echo "Plz write the filename:"
read filename

cd $filename

mkdir psi_blast
mkdir pssm

cd directory

for file in folder*;
do

	if [ ! -f ../Output/$i.psiblast ]; then
	psiblast -query $seq -db uniref90.db -num_iterations 3 -evalue 0.001 -out $seq.psiblast -out_ascii_pssm $seq.pssm -num_threads 8
	
	fi
	
done

mkdir psi_blast
mkdir pssm

mv *.psiblast psi_blast/
mv *.pssm pssm1/