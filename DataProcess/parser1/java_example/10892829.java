public class Test10892829 {
        @Override
        public InputStream getResourceAsStream(final String path) {
            final URL url = getResource(path);
            if (url != null) {
                try {
                    return AccessController.doPrivileged(new PrivilegedExceptionAction<InputStream>() {

                        public InputStream run() throws Exception {
                            try {
                                return url.openStream();
                            } catch (IOException e) {
                                LOG.warn("URL canot be accessed: " + e.getMessage());
                            }
                            return null;
                        }
                    }, m_accessControllerContext);
                } catch (PrivilegedActionException e) {
                    LOG.warn("Unauthorized access: " + e.getMessage());
                }
            }
            return null;
        }

}