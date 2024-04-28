from PyPDF2 import PdfReader

# Create a PdfFileReader object with the file
data_directory = "./content/"
pdf_reader = PdfReader(data_directory + "chat_docs4.pdf")


# Get the number of pages in the PDF
num_pages = len(pdf_reader.pages)
print( num_pages)

# Loop through each page and extract the text
for i in range(0, num_pages):
    page = pdf_reader.pages[i]
    text = page.extract_text()
    
    # Print or do something with the extracted text here
    # print("len:{} text:{}".format(len(text), text))

    filename = '{}/chat_docs{:3d}.txt'.format(data_directory, i)
    print( filename)
    with open(filename, 'w', encoding='utf-8') as f:
        lines = text.split('\n')
        non_blank_lines = [line for line in lines if line.strip() != '']
        f.write('\n'.join(non_blank_lines))
