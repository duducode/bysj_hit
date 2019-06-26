import tensorflow as tf
import matplotlib.pyplot as pt

#create a dataset that reads all of the examples from two files
filenames = tf.placeholder(tf.string,shape=[None])
dataset = tf.data.TFRecordDataset(filenames)

def get_one_sample(record):
    #next_element = iterator.get_next()

    features = {
        'image':tf.FixedLenFeature([],tf.string,default_value=""),
        'label':tf.FixedLenFeature([],tf.int64,default_value=tf.zeros([],dtype=tf.int64))
    }
    parsed =tf.parse_single_example(record,features)

    image = tf.decode_raw(parsed["image"],tf.uint8)
    image = tf.reshape(image,[112,112,1])
    newimage = tf.image.resize_images(image,(224,224))
    newimage =tf.reshape(newimage,[224,224])

    label =tf.cast(parsed["label"],tf.int32)
    return newimage,label

dataset = dataset.map(get_one_sample)
batched_dataset = dataset.batch(4)

iterator = batched_dataset.make_initializable_iterator()
sample = iterator.get_next()
#sample=get_one_sample(iterator)

sess = tf.Session()

train_file = ["train128.tfrecord"]
sess.run(iterator.initializer,feed_dict={filenames:train_file})

for i in range(128):
    value1,value2 = sess.run(sample)

#for i in range(0,4):
    #if i==3:
#        value = sess.run(newimage)
#print(value[51][30])
        #print(tf.shape(value))
    pt.imshow(value1[0],cmap=pt.cm.gray)
    pt.show()
        #value = sess.run(label)
    print(value2[0])
#value = sess.run(newimage)
#print(value[51][30])
#print(tf.shape(value))
#pt.imshow(value)
#pt.show()
#value = sess.run(label)
#print(value)
#validation_file = ["file1.tfrecord"]
#sess.run(iterator.initializer,feed_dict={filenames:validation_file})




