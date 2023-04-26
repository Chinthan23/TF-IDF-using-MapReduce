import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;
import java.util.*;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.StringUtils; 
import opennlp.tools.stemmer.PorterStemmer;

public class DocumentFreq {
	public static class FrequencyMapper extends Mapper<Object, Text, Text, Text> {

//		private final static IntWritable one = new IntWritable(1);
		private Text word = new Text();

		private boolean caseSensitive = false;
		private Set<String> patternsToSkip = new HashSet<String>();
		private PorterStemmer stemming= new PorterStemmer(); 

		@Override
		public void setup(Context context) throws IOException, InterruptedException {
			Configuration conf = context.getConfiguration();
			caseSensitive = conf.getBoolean("docfreq.case.sensitive", false);
			if (conf.getBoolean("docfreq.skip.patterns", false)) {
				URI[] patternsURIs = Job.getInstance(conf).getCacheFiles();
				for (URI patternsURI : patternsURIs) {
					Path patternsPath = new Path(patternsURI.getPath());
					String patternsFileName = patternsPath.getName().toString();
					parseSkipFile(patternsFileName);
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

		@Override
		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			String line = (caseSensitive) ? value.toString() : value.toString().toLowerCase();
			line=line.replaceAll("[^a-zA-Z ]","");
			String[] tokens = line.split("[^\\w']+");
			String fileName = ((FileSplit) context.getInputSplit()).getPath().getName();
			for (String token : tokens) {
				if(patternsToSkip.contains(token)) {
					continue;
				}
				String stemmed = stemming.stem(token);
				word.set(stemmed);
				context.write(word, new Text(fileName));
			}
		}
	}

	public static class IntSumReducer extends Reducer<Text, Text, Text, IntWritable> {
//		private IntWritable result = new IntWritable();
		private TreeMap<Text,Integer> top100=new TreeMap<Text,Integer>();

		public void reduce(Text key, Iterable<Text> values, Context context)
				throws IOException, InterruptedException {
			Set<String> articles=new HashSet<String>();
			for (Text val : values) {
				articles.add(val.toString());
			}
//			result.set(sum);
//			context.write(key,sum)
			top100.put(new Text(key.toString()),articles.size());
//			if(top100.size()>100) {
//				top100.remove(top100.firstKey());
//			}
		}
		
		@Override
		public void cleanup(Context context) throws IOException, InterruptedException{
			int i=0;
			TreeMap<Text,Integer> sorted=new TreeMap<Text,Integer>(valueComparator);
			sorted.putAll(top100);
			Map<Text,Integer> sorted_desc=sorted.descendingMap();
			for (Map.Entry<Text, Integer> entry: sorted_desc.entrySet()) {
				if(i==100) {
					break;
				}
				context.write(entry.getKey(), new IntWritable(entry.getValue()));
				i++;
			}
		}
		 public Comparator<Text> valueComparator = new Comparator<Text>() {
             public int compare(Text k1,Text k2)
             {
                 int comp = top100.get(k1).compareTo(
                     top100.get(k2));
                 if (comp == 0)
                     return 1;
                 else
                     return comp;
             }
       
         };
	}

	public static void main(String[] args) throws Exception {

		Configuration conf = new Configuration();
		Job job = Job.getInstance(conf, "docfreq");

		job.setMapperClass(FrequencyMapper.class);
		// job.setCombinerClass(IntSumReducer.class); // enable to use 'local aggregation'
		job.setReducerClass(IntSumReducer.class);

		job.setMapOutputValueClass(Text.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(IntWritable.class);
		job.getConfiguration().set("mapreduce.output.textoutputformat.separator", "\t");
		job.setOutputFormatClass(TextOutputFormat.class);
		for (int i = 0; i < args.length; ++i) {
			if ("-skippatterns".equals(args[i])) {
				job.getConfiguration().setBoolean("docfreq.skip.patterns", true);
				job.addCacheFile(new Path(args[++i]).toUri());
			} else if ("-casesensitive".equals(args[i])) {
				job.getConfiguration().setBoolean("docfreq.case.sensitive", true);
			}
		}

		FileInputFormat.addInputPath(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, new Path(args[1]));

		System.exit(job.waitForCompletion(true) ? 0 : 1);
	}
}
