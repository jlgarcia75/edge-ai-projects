from pandas_profiling import ProfileReport

def display_fancy(df, report_title, html_title):
    profile = ProfileReport(df, title=report_title)
    profile.to_file(html_title)
