package com.intel.sparkColumnarPlugin.vectorized;

import java.io.IOException;

public class ShuffleSplitterJniWrapper {

  public ShuffleSplitterJniWrapper() throws IOException {
    JniUtils.getInstance();
  }

  /**
   * Construct native splitter for shuffled RecordBatch over
   *
   * @param schemaBuf serialized arrow schema
   * @return native splitter instance id if created successfully.
   * @throws RuntimeException
   */
  public native long make(byte[] schemaBuf) throws RuntimeException;

  public native void split(long splitterId, int numRows, long[] bufAddrs, long[] bufSizes)
      throws RuntimeException;

  /**
   * Write the data remained in the buffers hold by native splitter to each partition's temporary
   * file. And stop processing splitting
   *
   * @param splitterId
   * @throws RuntimeException
   */
  public native void stop(long splitterId) throws RuntimeException;

  /**
   * Set the output buffer for each partition. Splitter will maintain one buffer for each partition
   * id occurred, and write data to file when buffer is full. Default buffer size will be set to 4096 rows.
   *
   * @param splitterId
   * @param bufferSize In row, not bytes. Default buffer size will be set to 4096 rows.
   */
  public native void setPartitionBufferSize(long splitterId, long bufferSize);

  /**
   * Set compression codec for splitter's output. For now we only support those types supported both by spark and arrow:
   * Default will be uncompressed.
   *
   * @param splitterId
   * @param codec "lz4", "snappy", "zstd", "uncompressed"
   */
  public native void setCompressionCodec(long splitterId, String codec);

  /**
   * Get all files information created by the splitter. Used by the {@link
   * org.apache.spark.shuffle.ColumnarShuffleWriter} These files are temporarily existed and will be
   * deleted after the combination.
   *
   * @param splitterId
   * @return an array of all files information
   */
  public native PartitionFileInfo[] getPartitionFileInfo(long splitterId);

  public native long getTotalBytesWritten(long splitterId);

  /**
   * Release resources associated with designated splitter instance.
   *
   * @param splitterId of the splitter instance.
   */
  public native void close(long splitterId);
}
