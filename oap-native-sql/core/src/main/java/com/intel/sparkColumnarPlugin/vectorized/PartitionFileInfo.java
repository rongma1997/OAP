package com.intel.sparkColumnarPlugin.vectorized;

public class PartitionFileInfo {
    private int pid;
    private String filePath;

    public PartitionFileInfo(int pid, String filePath) {
        this.pid = pid;
        this.filePath = filePath;
    }

    public int getPid() {
        return pid;
    }

    public String getFilePath() {
        return filePath;
    }
}
