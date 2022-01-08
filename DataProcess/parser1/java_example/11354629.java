public class Test11354629 {
    private Promise<Void> readyForWrite(final int limit) {
        if (bufferedData == null || bufferedData.length() <= limit) {
            return null;
        } else {
            if (inClosed) {
                throw new IllegalStateException("input is alread closed write cannot complete");
            }
            return new When<Void, Void>(writes.awaken()) {

                @Override
                public Promise<Void> resolved(Void o) {
                    return readyForWrite(limit);
                }
            }.promise();
        }
    }

}