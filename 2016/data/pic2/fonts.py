import matplotlib.font_manager as fm
import pandas as pd

fonts = fm.findSystemFonts()
# len(fonts) => 100以上はあるはず

l = []
for f in fonts:
    font = fm.FontProperties(fname=f)
    l.append((f, font.get_name(), font.get_family()))
df = pd.DataFrame(l, columns=['path', 'name', 'family'])

df[df['path'].apply(lambda s: 'IPA' in s)]
