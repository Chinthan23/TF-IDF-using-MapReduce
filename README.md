# Running the code
Setup hadoop on your system.
Export HADOOP_HOME as the location of your hadoop installation.
Export PATH as HADOOP_HOME/bin:HADOOP_HOME/sbin.
Export classpath as HADOOP_CLASSPATH to reference your open nlp classpath.

To run any of the jars of problem 1 use the command:
	- hadoop jar JarFile.jar MainEntryPoint ./input_directory ./output_directory
If you want to check the performance add time as prefix.

To compute IDF for problem 2:
	- hadoop jar DocFreq.jar DocumentFreq ./input_directory ./output_directory-df -skippatterns stopwords.txt

Then rename the output file with .tsv as it is a tsv file.
	- mv ./output_directory/part-r-00000 ./output_directory-df/part-r-00000.tsv

Use this as input for IDF:
	- hadoop jar IDF.jar IDF ./input_directory ./output_directory -skippatterns stopwords.txt -tsv ./output_directory-df/part-r-00000.tsv

We can check the outputs in output_directories.
