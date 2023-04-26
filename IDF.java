import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;
import java.util.*;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.MapWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.StringUtils; 
import opennlp.tools.stemmer.PorterStemmer;

public class IDF {
	public static class FrequencyMapper extends Mapper<Object, Text, Text, MapWritable> {
		private Text word = new Text();

		private boolean caseSensitive = false;
		public HashMap<String,Integer> df_from_tsv=new HashMap<String,Integer>();
		private Set<String> patternsToSkip = new HashSet<String>();
		private PorterStemmer stemming= new PorterStemmer(); 
		@Override
		public void setup(Context context) throws IOException, InterruptedException {
			Configuration conf = context.getConfiguration();
			caseSensitive = conf.getBoolean("idf.case.sensitive", false);
			URI[] URIs = Job.getInstance(conf).getCacheFiles();
			for (URI URI : URIs) {
				Path path = new Path(URI.getPath());
				String fileName = path.getName().toString();
				if(fileName.endsWith(".txt") && conf.getBoolean("idf.skip.patterns", false)) {						
					parseSkipFile(fileName);
				}
				else if(fileName.endsWith(".tsv") && conf.getBoolean("idf.tsv", false)) {
					createMapFromTSV(fileName);
				}
			}
		}

		private void parseSkipFile(String fileName) {
			try {
				BufferedReader reader = new BufferedReader(new FileReader(fileName));
				String pattern = null;
				while ((pattern = reader.readLine()) != null) {
					patternsToSkip.add(pattern);
				}
				reader.close();
			} catch (IOException ioe) {
				System.err.println(
						"Caught exception while parsing the cached file '" + StringUtils.stringifyException(ioe));
			}
		}
		private void createMapFromTSV(String fileName) {
			try {
				BufferedReader reader = new BufferedReader(new FileReader(fileName));
				String line;
				while ((line = reader.readLine()) != null) {
					String[] row=line.split("\t");
					if(row.length==2) {
						String key =row[0];
						Integer value= Integer.parseInt(row[1]);
						df_from_tsv.put(key, value);
					}
				}
				reader.close();
			} catch (IOException ioe) {
				System.err.println(
						"Caught exception while parsing the cached file '" + StringUtils.stringifyException(ioe));
			}
		}

		@Override
		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			String line = (caseSensitive) ? value.toString() : value.toString().toLowerCase();
			String fileName = ((FileSplit) context.getInputSplit()).getPath().getName();
			line=line.replaceAll("[^a-zA-Z ]","");
			String[] tokens = line.split("[^\\w']+");
			MapWritable strips=new MapWritable();
			for (String token : tokens) {
				if(patternsToSkip.contains(token)) {
					continue;
				}
				String stemmed = stemming.stem(token);
				if(df_from_tsv.containsKey(stemmed)) {
					Text stemmed_word=new Text(stemmed);
					if(strips.containsKey(stemmed_word)) {
						int count=((IntWritable)strips.get(stemmed_word)).get();
						strips.put(stemmed_word, new IntWritable(count+1));
					}
					else {
						strips.put(stemmed_word, new IntWritable(1));
					}
				}
				word.set(fileName);
				context.write(word, strips);
			}
		}
	}

	public static class IntSumReducer extends Reducer<Text, MapWritable, Text, DoubleWritable> {
		private Text output_key=new Text();
		public HashMap<String,Integer> df_from_tsv=new HashMap<String,Integer>();
		public void setup(Context context) throws IOException, InterruptedException {
			Configuration conf = context.getConfiguration();
			URI[] URIs = Job.getInstance(conf).getCacheFiles();
			for (URI URI : URIs) {
				Path path = new Path(URI.getPath());
				String fileName = path.getName().toString();
				if(fileName.endsWith(".tsv") && conf.getBoolean("idf.tsv", false)) {
					createMapFromTSV(fileName);
				}
			}
		}
		private void createMapFromTSV(String fileName) {
			try {
				BufferedReader reader = new BufferedReader(new FileReader(fileName));
				String line;
				while ((line = reader.readLine()) != null) {
					String[] row=line.split("\t");
					if(row.length==2) {
						String key =row[0];
						Integer value= Integer.parseInt(row[1]);
						df_from_tsv.put(key, value);
					}
				}
				reader.close();
			} catch (IOException ioe) {
				System.err.println(
						"Caught exception while parsing the cached file '" + StringUtils.stringifyException(ioe));
			}
		}
		@Override
		public void reduce(Text key, Iterable<MapWritable> values, Context context)
				throws IOException, InterruptedException {
			MapWritable last_associative_array= new MapWritable();
			for (MapWritable array : values) {
				for(Writable keys: array.keySet()) {
					if(last_associative_array.containsKey(keys)) {
						int to_add=((IntWritable) array.get(keys)).get();
						int count=((IntWritable) last_associative_array.get(keys)).get();
						last_associative_array.put((Text) keys,new IntWritable(count+to_add));
					}
					else {
						last_associative_array.put((Text) keys,(IntWritable) array.get(keys));
					}
				}
			}
			String filename=key.toString();
			String key_with_file=filename+"\t";
			
			for(Writable term: last_associative_array.keySet()) {
				String key_out=key_with_file+term.toString();
				int tf=((IntWritable)last_associative_array.get(term)).get();
				int df=df_from_tsv.getOrDefault(term.toString(),0);
				double score=tf*Math.log(10000.0 / (df+1));
				output_key.set(key_out);
				context.write(output_key, new DoubleWritable(score));
			}
		}
	}

	public static void main(String[] args) throws Exception {

		Configuration conf = new Configuration();
		Job job = Job.getInstance(conf, "idf");

		job.setMapperClass(FrequencyMapper.class);
		// job.setCombinerClass(IntSumReducer.class); // enable to use 'local aggregation'
		job.setReducerClass(IntSumReducer.class);
		
		job.setMapOutputKeyClass(Text.class);
		job.setMapOutputValueClass(MapWritable.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(DoubleWritable.class);
		
		job.getConfiguration().set("mapreduce.output.textoutputformat.separator", "\t");
		job.setOutputFormatClass(TextOutputFormat.class);
		
		for (int i = 0; i < args.length; ++i) {
			if ("-skippatterns".equals(args[i])) {
				job.getConfiguration().setBoolean("idf.skip.patterns", true);
				job.addCacheFile(new Path(args[++i]).toUri());
			} else if ("-casesensitive".equals(args[i])) {
				job.getConfiguration().setBoolean("idf.case.sensitive", true);
			}
			else if("-tsv".equals(args[i])) {
				job.getConfiguration().setBoolean("idf.tsv",true);
				job.addCacheFile(new Path(args[++i]).toUri());
			}
		}

		FileInputFormat.addInputPath(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, new Path(args[1]));

		System.exit(job.waitForCompletion(true) ? 0 : 1);
	}
}
