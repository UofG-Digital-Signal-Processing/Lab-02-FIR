# Submission

Naming convention for the files in the zip (all files lower caps and in a subdirectory containing your matric numbers):

- ecg.dat 
- firfilter.py 
- hpbsfilter.py 
- lmsfilter.py 
- hrdetect.py 
- report.pdf

firfilter.py is the FIR filter module containing the FIRfilter class.

hpbsfilter.py is the main program which uses firfilter to remove DC and 50Hz (1-3). 

lmsfilter.py is the main program which uses an LMS filter to remove 50Hz (4). 

hrdetect.py is the R peak and heartrate detector and must show a graph of the momentary heartrate against time (5).

All files need to be put into a single subdirectory with your matric numbers in it called fir_12234567a_12345886b and then zipped into a single zip with the same name: fir_12234567a_12345886b.zip. Make sure that both python files execute directly from the command line, for example by typing "python3 hr_detect.py".

