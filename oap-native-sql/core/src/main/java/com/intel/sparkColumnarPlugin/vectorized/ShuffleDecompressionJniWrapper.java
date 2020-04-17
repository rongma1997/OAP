package com.intel.sparkColumnarPlugin.vectorized;

import java.io.IOException;

public class ShuffleDecompressionJniWrapper {

  public ShuffleDecompressionJniWrapper() throws IOException {
    JniUtils.getInstance();
  }

  /**
   * Make for multiple decompression with the same schema
   *
   * @param schemaBuf serialized arrow schema
   * @return native schema holder id
   * @throws RuntimeException
   */
  public native long make(byte[] schemaBuf) throws RuntimeException;

  public native ArrowRecordBatchBuilder decompress(
      long schemaHolderId, String compressionCodec, int numRows, long[] bufAddrs, long[] bufSizes, long[] bufMask)
      throws RuntimeException;

  /**
   * Release resources associated with designated schema holder instance.
   *
   * @param schemaHolderId of the schema holder instance.
   */
  public native void close(long schemaHolderId);
}
