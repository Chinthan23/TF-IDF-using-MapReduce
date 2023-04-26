
import java.io.IOException;
import java.io.File;

import opennlp.tools.cmdline.postag.POSModelLoader;
import opennlp.tools.postag.POSModel;
import opennlp.tools.postag.POSTaggerME;
import opennlp.tools.tokenize.SimpleTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.MapWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class POSstripes {
		private static POSModel model= new POSModelLoader().load(new File("/Users/chinthanchandra/Desktop/IIITB/3rd_Year/sem6/nosql/A2/opennlp-en-ud-ewt-pos-1.0-1.9.3.bin"));
		private static POSTaggerME tagger= new POSTaggerME(model);
		private static SimpleTokenizer tokenizer= SimpleTokenizer.INSTANCE;
		
		public static class POSTaggerAndMapper extends Mapper<Object, Text, Text, MapWritable> {

//			private final static IntWritable one = new IntWritable(1);
			@Override
			public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
				String articles = value.toString();
				String tokenizedArticle[]=tokenizer.tokenize(articles);
				String[] tags= tagger.tag(tokenizedArticle);
				MapWritable stripes=new MapWritable();
				
				for (String tag: tags) {
					Text posTag=new Text(tag);
					if(stripes.containsKey(posTag)){
						int count=((IntWritable)stripes.get(posTag)).get();
						stripes.put(posTag, new IntWritable(count+1));
					}
					else {
						stripes.put(posTag,new IntWritable(1));
					}
					
				}
				context.write(new Text("0"), stripes);
			}
		}

		public static class IntSumReducer extends Reducer<Text, MapWritable, Text, IntWritable> {
			private IntWritable result = new IntWritable();

			@Override
			public void reduce(Text key, Iterable<MapWritable> values, Context context)
					throws IOException, InterruptedException {
				MapWritable total_count= new MapWritable();
				for (MapWritable val : values) {
					for(Writable keys: val.keySet()) {
						if(total_count.containsKey(keys)) {
							int to_add=((IntWritable)val.get(keys)).get();
							int count=((IntWritable)total_count.get(keys)).get();
							total_count.put((Text)keys, new IntWritable(count+to_add));
						}
						else {
							total_count.put((Text)keys,(IntWritable)val.get(keys));
						}
					}
				}
				for(Writable key_in_total_count: total_count.keySet()) {
					result.set(((IntWritable)total_count.get(key_in_total_count)).get());					
					context.write((Text)key_in_total_count, result);
				}
			}

		}

		public static void main(String[] args) throws Exception {

			Configuration conf = new Configuration();
			Job job = Job.getInstance(conf, "postag");

			job.setMapperClass(POSTaggerAndMapper.class);
			// job.setCombinerClass(IntSumReducer.class); // enable to use 'local aggregation'
			job.setReducerClass(IntSumReducer.class);

			job.setMapOutputKeyClass(Text.class);
			job.setMapOutputValueClass(MapWritable.class);
			job.setOutputKeyClass(Text.class);
			job.setOutputValueClass(IntWritable.class);

			FileInputFormat.addInputPath(job, new Path(args[0]));
			FileOutputFormat.setOutputPath(job, new Path(args[1]));

			System.exit(job.waitForCompletion(true) ? 0 : 1);
		}
}
