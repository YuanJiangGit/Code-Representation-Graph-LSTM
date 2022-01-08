public class Test3870843 {
            protected void readImpl(PipelineContext context, ContentHandler contentHandler) {
                try {
                    Schema schema = (Schema) readCacheInputAsObject(context, getInputByName(INPUT_SCHEMA), new CacheableInputReader() {

                        public Object read(PipelineContext context, ProcessorInput input) {
                            try {
                                long time = 0;
                                if (logger.isDebugEnabled()) {
                                    logger.debug("Reading Schema: " + schemaId);
                                    time = System.currentTimeMillis();
                                }
                                Document schemaDoc = readInputAsDOM4J(context, input);
                                LocationData locator = (LocationData) schemaDoc.getRootElement().getData();
                                final String schemaSystemId = (locator != null && locator.getSystemID() != null) ? locator.getSystemID() : null;
                                VerifierFactory verifierFactory = new TheFactoryImpl(getFactory());
                                verifierFactory.setEntityResolver(new EntityResolver() {

                                    public InputSource resolveEntity(String publicId, String systemId) throws IOException {
                                        URL url = URLFactory.createURL(schemaSystemId, systemId);
                                        InputSource i = new InputSource(url.openStream());
                                        i.setSystemId(url.toString());
                                        return i;
                                    }
                                });
                                InputSource is = new InputSource(new StringReader(XMLUtils.domToString(schemaDoc)));
                                is.setSystemId(schemaSystemId);
                                synchronized (MSVValidationProcessor.class) {
                                    Schema schema = verifierFactory.compileSchema(is);
                                    if (logger.isDebugEnabled()) logger.debug(schemaId + " : Schema compiled in " + (System.currentTimeMillis() - time));
                                    return schema;
                                }
                            } catch (VerifierConfigurationException vce) {
                                throw new OXFException(vce.getCauseException());
                            } catch (Exception e) {
                                throw new OXFException(e);
                            }
                        }
                    });
                    final Verifier verifier = schema.newVerifier();
                    abstract class Handler extends ForwardingContentHandler implements ErrorHandler {

                        protected Handler(ContentHandler contentHandler) {
                            super(contentHandler);
                        }
                    }
                    ;
                    final ContentHandler _contentHandler = contentHandler;
                    Handler stateContentHandler = new Handler(verifier.getVerifierHandler()) {

                        public boolean recording;

                        private String uri;

                        private String localname;

                        private String qName;

                        private Attributes attributes;

                        private SAXStore saxStore = new SAXStore();

                        private Stack relevantState = new Stack();

                        private boolean inAnnotateState;

                        public void startDocument() throws SAXException {
                            super.startDocument();
                            _contentHandler.startDocument();
                        }

                        public void endDocument() throws SAXException {
                            super.endDocument();
                            _contentHandler.endDocument();
                        }

                        public void startElement(String uri, String localname, String qName, Attributes attributes) throws SAXException {
                            flushRecordIfNeeded();
                            this.uri = uri;
                            this.localname = localname;
                            this.qName = qName;
                            this.attributes = new AttributesImpl(attributes);
                            this.recording = true;
                            {
                                int index = attributes.getIndex(Constants.XXFORMS_NAMESPACE_URI, "relevant");
                                relevantState.push((index != -1 && "false".equals(attributes.getValue(index))) ? null : this);
                            }
                            super.startElement(uri, localname, qName, attributes);
                        }

                        public void endElement(String uri, String localname, String qName) throws SAXException {
                            inAnnotateState = true;
                            super.endElement(uri, localname, qName);
                            inAnnotateState = false;
                            flushRecordIfNeeded();
                            _contentHandler.endElement(uri, localname, qName);
                            relevantState.pop();
                        }

                        public void characters(char[] chars, int start, int length) throws SAXException {
                            inAnnotateState = true;
                            super.characters(chars, start, length);
                            inAnnotateState = false;
                            if (recording) {
                                saxStore.characters(chars, start, length);
                            } else {
                                _contentHandler.characters(chars, start, length);
                            }
                        }

                        public void startPrefixMapping(String s, String s1) throws SAXException {
                            super.startPrefixMapping(s, s1);
                            if (recording) {
                                saxStore.startPrefixMapping(s, s1);
                            } else {
                                _contentHandler.startPrefixMapping(s, s1);
                            }
                        }

                        public void endPrefixMapping(String s) throws SAXException {
                            super.endPrefixMapping(s);
                            if (recording) {
                                saxStore.endPrefixMapping(s);
                            } else {
                                _contentHandler.endPrefixMapping(s);
                            }
                        }

                        public void processingInstruction(String s, String s1) throws SAXException {
                            super.processingInstruction(s, s1);
                            if (recording) {
                                saxStore.processingInstruction(s, s1);
                            } else {
                                _contentHandler.processingInstruction(s, s1);
                            }
                        }

                        public void ignorableWhitespace(char[] chars, int start, int length) throws SAXException {
                            super.ignorableWhitespace(chars, start, length);
                            if (recording) {
                                saxStore.ignorableWhitespace(chars, start, length);
                            } else {
                                _contentHandler.ignorableWhitespace(chars, start, length);
                            }
                        }

                        public void skippedEntity(String s) throws SAXException {
                            super.skippedEntity(s);
                            if (recording) {
                                saxStore.skippedEntity(s);
                            } else {
                                _contentHandler.skippedEntity(s);
                            }
                        }

                        private void flushRecordIfNeeded() throws SAXException {
                            if (recording) {
                                _contentHandler.startElement(uri, localname, qName, attributes);
                                saxStore.replay(_contentHandler);
                                saxStore.clear();
                                recording = false;
                            }
                        }

                        private void generateErrorAnnotation(ValidationException exception) {
                            if (recording && inAnnotateState && isCurrentElementRelevant()) {
                                AttributesImpl newAttributes = new AttributesImpl(attributes);
                                newAttributes.addAttribute(Constants.XXFORMS_NAMESPACE_URI, Constants.XXFORMS_ERROR_ATTRIBUTE_NAME, Constants.XXFORMS_PREFIX + ":" + Constants.XXFORMS_ERROR_ATTRIBUTE_NAME, "CDATA", exception.getMessage());
                                newAttributes.addAttribute(Constants.XXFORMS_NAMESPACE_URI, Constants.XXFORMS_VALID_ATTRIBUTE_NAME, Constants.XXFORMS_PREFIX + ":" + Constants.XXFORMS_VALID_ATTRIBUTE_NAME, "CDATA", "false");
                                attributes = newAttributes;
                            }
                        }

                        private boolean isCurrentElementRelevant() {
                            for (Iterator i = relevantState.iterator(); i.hasNext(); ) {
                                if (i.next() == null) return false;
                            }
                            return true;
                        }

                        public void error(SAXParseException exception) {
                            generateErrorAnnotation(new ValidationException("Warning " + exception.getMessage() + "(schema: " + schemaId + ")", new LocationData(exception)));
                        }

                        public void fatalError(SAXParseException exception) {
                            generateErrorAnnotation(new ValidationException("Fatal Error " + exception.getMessage() + "(schema: " + schemaId + ")", new LocationData(exception)));
                        }

                        public void warning(SAXParseException exception) {
                            generateErrorAnnotation(new ValidationException("Warning " + exception.getMessage() + "(schema: " + schemaId + ")", new LocationData(exception)));
                        }
                    };
                    verifier.setErrorHandler(stateContentHandler);
                    long time = 0;
                    if (logger.isDebugEnabled()) {
                        time = System.currentTimeMillis();
                    }
                    readInputAsSAX(context, getInputByName(INPUT_DATA), stateContentHandler);
                    if (logger.isDebugEnabled()) {
                        logger.debug(schemaId + " validation completed in " + (System.currentTimeMillis() - time));
                    }
                } catch (Exception e) {
                    throw new OXFException(e);
                }
            }

}