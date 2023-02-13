for line in $(<images.txt) 
do
    wget "https://download.brainlib.org/hackathon/2022_GYBS/input/fMOST/subject/$(echo $line | tr -d '\r')"
done 

