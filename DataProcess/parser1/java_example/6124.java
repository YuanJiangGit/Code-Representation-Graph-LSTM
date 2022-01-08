public class Test6124 {
            public static boolean onEvent(SKJMessage msg) {
                if (msg instanceof SK_RequestChannelAck) {
                    SK_RequestChannelAck ack = (SK_RequestChannelAck) msg;
                    if (ack.getSKStatus() == SKConstants.OK) {
                        logger.info("Got channel " + ack.getSpan() + ":" + ack.getChannel());
                        byte COLLECT_FIXED_NO_OF_DIGITS = 0x01;
                        byte COLLECT_MAX_DIGITS = 1;
                        byte COLLECT_NUM_TERM_CHARS = 0;
                        byte COLLECT_CONFIG_BITS = 0x1E;
                        int COLLECT_INTER_DIGIT_TIMER = 5000;
                        int COLLECT_FIRST_DIGIT_TIMER = 5000;
                        int COLLECT_COMPLETION_TIMER = 6000;
                        int COLLECT_MIN_RECEIVE_DIGIT_DURATION = 5;
                        byte COLLECT_NUM_DIGIT_STRINGS = 1;
                        int COLLECT_RESUME_DIGIT_CLTN_TIMER = 0;
                        int DSP_SERVICE_TYPE_DTMF_RECEIVER = (byte) 0x01;
                        int serviceType = DSP_SERVICE_TYPE_DTMF_RECEIVER;
                        XL_CollectDigitString cd = new XL_CollectDigitString();
                        cd.setNodeID(0xff);
                        cd.setSpan(ack.getSpan());
                        cd.setChannel(ack.getChannel());
                        cd.setMode(COLLECT_FIXED_NO_OF_DIGITS);
                        cd.setMaxDigits(COLLECT_MAX_DIGITS);
                        cd.setNumTermChars(COLLECT_NUM_TERM_CHARS);
                        cd.setConfigBits(COLLECT_CONFIG_BITS);
                        cd.setInterDigitTimer(COLLECT_INTER_DIGIT_TIMER);
                        cd.setFirstDigitTimer(COLLECT_FIRST_DIGIT_TIMER);
                        cd.setCompletionTimer(COLLECT_COMPLETION_TIMER);
                        cd.setMinReceiveDigitDuration(COLLECT_MIN_RECEIVE_DIGIT_DURATION);
                        cd.setAddressSignallingType(serviceType);
                        cd.setNumDigitStrings(COLLECT_NUM_DIGIT_STRINGS);
                        cd.setResumeDigitCltnTimer(COLLECT_RESUME_DIGIT_CLTN_TIMER);
                        class CollectDigitStringAckListener implements SKJEventListener {

                            public boolean onEvent(SKJMessage skjmessage) {
                                logger.log(Level.ALL, "entering");
                                boolean isProcessed = false;
                                if (skjmessage instanceof XL_CollectDigitStringAck) {
                                    isProcessed = true;
                                    System.out.println("Got the ACK!");
                                }
                                logger.log(Level.ALL, "exiting");
                                return isProcessed;
                            }
                        }
                        CollectDigitStringAckListener msgListener = new CollectDigitStringAckListener();
                        csp.sendMessage(cd, msgListener);
                        return true;
                    } else {
                        logger.info("Request channel got status:" + SKJava.statusText(ack.getSKStatus()));
                    }
                }
                return false;
            }

}