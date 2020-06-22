package com.intel.sparkColumnarPlugin.vectorized;

import java.io.IOException;

public class UtilsJniWrapper {

  public UtilsJniWrapper() throws IOException {
    JniUtils.getInstance();
  }

  /**
   * get timestamp in micro second from native
   *
   * @return timestamp 
   * @throws RuntimeException
   */
  public native long getTime() throws RuntimeException;

}
