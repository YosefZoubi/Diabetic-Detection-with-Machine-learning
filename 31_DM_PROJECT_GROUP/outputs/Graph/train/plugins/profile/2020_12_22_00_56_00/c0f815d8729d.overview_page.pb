?	?o?l?@?o?l?@!?o?l?@	?C???????C??????!?C??????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?o?l?@????@1~???UH?@A??.?u???Iut\?l??Y?e?؊??*	??/?[?@2K
Iterator::Model::Map???l???!v?uk9?W@)?'HlwO??1??U$$(W@:Preprocessing2q
:Iterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat?2?Pl??!8?<EU5	@)62;?ޡ?1???m@:Preprocessing2Z
#Iterator::Model::Map::ParallelMapV2w?k?F=??!???QE??)w?k?F=??1???QE??:Preprocessing2_
(Iterator::Model::Map::ParallelMapV2::Zip8k??*??!yw????@)(b?c??1??????:Preprocessing2?
TIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???hqƀ?!?21???)???hqƀ?1?21???:Preprocessing2{
DIterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?x]???! 8?VZ??)]N	?I???14??{???:Preprocessing2}
FIterator::Model::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?lɪ7y?!????<??)?lɪ7y?1????<??:Preprocessing2F
Iterator::Model??m4????!?xw???W@)`=?[?w?1??i???:Preprocessing2k
4Iterator::Model::Map::ParallelMapV2::Zip[0]::FlatMap-?"?J ??!?#??U??)>????q?1C???? ??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?C??????I ?pC????Qz?????X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????@????@!????@      ??!       "	~???UH?@~???UH?@!~???UH?@*      ??!       2	??.?u?????.?u???!??.?u???:	ut\?l??ut\?l??!ut\?l??B      ??!       J	?e?؊???e?؊??!?e?؊??R      ??!       Z	?e?؊???e?؊??!?e?؊??b      ??!       JGPUY?C??????b q ?pC????yz?????X@?"R
3DR-Model-Based-VGG16/block1_conv2/Relu:_FusedConv2DUnknown"4??t???!"4??t???0"R
3DR-Model-Based-VGG16/block2_conv2/Relu:_FusedConv2DUnknown?9	?????! ?t??&??0"R
3DR-Model-Based-VGG16/block3_conv2/Relu:_FusedConv2DUnknownV9????!V?a5?G??0"R
3DR-Model-Based-VGG16/block3_conv3/Relu:_FusedConv2DUnknown6???3??!d????T??0"R
3DR-Model-Based-VGG16/block2_conv1/Relu:_FusedConv2DUnknown
G_-?ճ?!?^1/%??0"R
3DR-Model-Based-VGG16/block4_conv3/Relu:_FusedConv2DUnknown3???????!??1JM=??0"R
3DR-Model-Based-VGG16/block4_conv2/Relu:_FusedConv2DUnknown\??߄??!dP?=?M??0"-
IteratorGetNext/_1_SendVw?ʉa??!?g[???"R
3DR-Model-Based-VGG16/block3_conv1/Relu:_FusedConv2DUnknownf???????!?e????0"R
3DR-Model-Based-VGG16/block4_conv1/Relu:_FusedConv2DUnknown??????!'???????0Q      Y@Y???=a3@a
{?'T@qd?C? ??y?v??)?:?"?	
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 