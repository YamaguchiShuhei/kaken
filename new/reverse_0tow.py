def reverse(text, label, b, m, e, s, char_id):
    raw_text = []
    for i in text[0]:
        for char in char_id:
            if i == char_id[char]:
                raw_text.append(char)
    print(''.join(raw_text))
    for i in zip(raw_text,label[0],b[0],m[0],e[0],s[0]):
        print(i)
