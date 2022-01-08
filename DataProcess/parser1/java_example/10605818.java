public class Test10605818 {
    private void template2sql(File f) throws Exception {
        FileInputStream fis = new FileInputStream(f);
        SqlExecutor executor = new SimpleSqlExecutor(connection);
        cntFile++;
        try {
            byte[] buf = new byte[fis.available()];
            fis.read(buf);
            Map<String, String> param = new HashMap<String, String>();
            String name = getName(f);
            if (StringUtils.isNotEmpty(prefix)) {
                name = prefix + name;
            }
            param.put("name", name);
            Locale locale = LocaleUtils.localeFromFileName(f, defaultLocale);
            param.put("language", getLocaleValue(locale.getLanguage()));
            param.put("country", getLocaleValue(locale.getCountry()));
            param.put("variant", getLocaleValue(locale.getVariant()));
            String source;
            if (StringUtils.isNotEmpty(charset)) {
                source = new String(buf, charset);
            } else {
                String c = LocaleUtils.charsetFromLocale(locale);
                if (StringUtils.isNotEmpty(c)) {
                    source = new String(buf, c);
                } else {
                    source = new String(buf);
                }
            }
            if (source.length() > 0 && source.charAt(0) == '﻿') {
                source = source.substring(1);
            }
            param.put("source", source);
            if (StringUtils.isNotEmpty(deleteSql)) {
                cntDel += executor.executeUpdate(deleteSql, param);
            }
            int cu = 0;
            if (StringUtils.isNotEmpty(updateSql)) {
                cu = executor.executeUpdate(updateSql, param);
                cntUpd += cu;
            }
            if (cu == 0) {
                cntIns += executor.executeUpdate(insertSql, param);
            }
            connection.commit();
        } catch (Exception e) {
            rollback();
            System.err.println("Failed to process " + f.getPath());
            throw e;
        } finally {
            IOUtils.closeQuietly(fis);
        }
    }

}