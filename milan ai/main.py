from asyncio import run
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from streamlit_extras.add_vertical_space import add_vertical_space
from io import BytesIO
import os
import re
from playwright.sync_api import sync_playwright
from playwright.sync_api import sync_playwright, Playwright
import asyncio
from playwright.async_api import async_playwright

import time

# Sidebar contents
with st.sidebar:
    logo = 'milan transparent.png'
    st.image(logo, width=300)
    st.title('Milan AI')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built to automate Data Entry.
    ''')
    add_vertical_space(5)
    st.write('')


os.environ["OPENAI_API_KEY"] = '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'
openai_api_key = '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'
openai_instance = OpenAI(openai_api_key=openai_api_key)


def select_box():
    option = st.selectbox(
    "Select the name of the Party",
    ("SHRI KRISHNA DELHI", "BHAIRAV", "VIPUL SURAT"),
    index=None,
    placeholder="Select party...",)
    return option

option = select_box()


async def ske_go():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, slow_mo=50)
        page = await browser.new_page()
        await page.goto("https://milansoft.co.in/")
        await page.click('input[placeholder="Username"]')
        await page.type('input[placeholder="Username"]', "admin")
        await page.click('input[placeholder="Password"]')
        await page.type('input[placeholder="Password"]', "harsh@2001")
        await page.get_by_role("button", name="Submit").click()
        await page.wait_for_load_state("load")
        await asyncio.sleep(2)
        await page.get_by_role("link", name=" PURCHASE ").dblclick()
        await page.get_by_role("link", name=" PURCHASE ").click()
        await asyncio.sleep(2)
        await page.get_by_role("link", name="PURCHASE INVOICE (MR)").click()
        await asyncio.sleep(2)
        await page.locator("#partyname").click()
        await page.locator("#partyname").type("Shri krishna")
        await asyncio.sleep(2)
        await page.locator("#partyindexkey_2").get_by_text("SHRI KRISHNA EMBROIDERIES PVT").click()
        await page.locator("#invno").click()
        await page.locator("#invno").type(f"{invoice_number}")
        await page.locator("#invno").type("123456")
        await page.locator("#invno").press("Tab")
        await page.locator("#invdate").press("Tab")
        await page.locator("#descript").press("Tab")
        await page.locator("#note").press("Tab")
        await page.locator("#issueto").press("Tab")
        # await page.locator("#itemcode3").type("121321")
        for hsn_code, quantity, description, edition_number, ske_code, price in zip(hsn_codes, qty_purchased, stripped_product_description, edition, ske_codes, prices): 
            await page.keyboard.type(f'{hsn_code}', delay=200)
            await asyncio.sleep(2)
            await page.keyboard.press('ArrowDown')
            await page.keyboard.press('Enter')
            await page.keyboard.press('Tab')
            await page.keyboard.type('Tissue')
            await page.keyboard.press('Tab')
            await page.keyboard.type('Org')
            await page.keyboard.press('Tab')
            await page.keyboard.type('SKE', delay=200)
            await page.keyboard.press('Tab')
            await page.keyboard.press('Tab')
            await page.keyboard.type(f'{edition_number}')
            await page.keyboard.press('Tab')
            await page.keyboard.type(f'{ske_code}')
            await page.keyboard.press('Tab')
            await page.keyboard.press('Tab')
            await page.keyboard.press('Tab')
            await page.keyboard.press('Tab')
            await page.keyboard.type(f'{quantity}')
            await page.keyboard.press('Tab')
            await page.keyboard.press('Tab')
            await page.keyboard.type(f'{price}')
            await page.keyboard.press('Tab')
            await page.keyboard.press('Tab')
            await page.keyboard.press('Tab')
            await page.keyboard.press('Tab')
            await page.keyboard.press('Tab')
            await page.keyboard.press('Tab')
            await page.keyboard.press('Tab')
            await page.keyboard.press('Tab')


            await asyncio.sleep(2)
            


            
        await asyncio.sleep(10)
        await browser.close()


if option == 'SHRI KRISHNA DELHI':

    st.header("Upload PDF 💬")
    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:

        st.write("The PDF Contains the following data")
        pdf_bytes = pdf.read()  # Read the raw content of the PDF file
        pdf_stream = BytesIO(pdf_bytes)  # Wrap it in a BytesIO object
        pdf_reader = PdfReader(pdf_stream)  # Pass the BytesIO object to PdfReader

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectorstore = FAISS.from_texts(chunks, embeddings)

        chain = load_qa_chain(openai_instance, chain_type="stuff")


        # Party Name
        query_party_name = "From The Text - JUST mention the name of the seller"
        docs_party = vectorstore.similarity_search(query_party_name)
        party_name = chain.run(input_documents=docs_party, question=query_party_name)
        st.write(f"Party Name: ",{party_name})

        
        
        # Invoice Number
        query_invoice_number = "From The Text - just mention the Invoice Number of the bill"
        docs_party = vectorstore.similarity_search(query_invoice_number)
        invoice_number = chain.run(input_documents=docs_party, question=query_invoice_number)
        st.write(f"Invoice Number: {invoice_number}")
        invoice_number_pattern = r"\b\d{4}\b"
        invoice_number = re.findall(invoice_number_pattern, invoice_number)
        # st.write(f"Last 4 digits - ",{invoice_number})


        # All the Products in the bill
        query_products = "LIST from the text in the following WAY - design number - item - hsn - rate - qty - UOM - value. GIVE IT TO ME AS A LIST"
        docs_product_list = vectorstore.similarity_search(query_products)
        # st.write(chain.run(input_documents=docs_product_list, question=query_products))
        product_list = []
        product_list = chain.run(input_documents=docs_product_list, question=query_products)
        text = product_list
        st.write(product_list)
            

        
        #SKE Codes - Design Number
        ske_pattern = r'SKE\d+'
        ske_codes = []
        ske_codes = re.findall(ske_pattern, text)
        # st.write(ske_codes)


        #HSN Codes
        hsn_pattern = r'\b\d{6}\b'
        hsn_codes = []
        hsn_codes = re.findall(hsn_pattern,text)
        # st.write(hsn_codes)
        # print(hsn_codes)


        #Quantity Purchased
        qty_pattern = r'\d{1,3} PCS|\d{1,3} - PCS'
        qty_purchased = []
        qty_purchased = re.findall(qty_pattern, text)
        # st.write(qty_purchased)
        # print(qty_purchased)


        # Product Description --- just the description
        product_description_pattern = r'- .*? \d+ MIX FABRIC SET -'
        product_description = []
        product_description =  re.findall(product_description_pattern, text)
        stripped_product_description = [strip[1:-1] for strip in product_description]
        # st.write(stripped_product_description)
        # print(stripped_product_description)


        # Edition Number
        product_description_string = ''.join(product_description) # Take the product description list and convert it into a string to run 4 dighit regex
        # print(product_description_string)
        edition_pattern = r'\b\d{4}\b'
        edition = []
        edition = re.findall(edition_pattern, product_description_string)
        # st.write(edition)
        # print(edition)
        #from product description finding fabric - top & dupatta
        st.write(product_description_string)
        # fabric_top = 


        #Extraction of Price
        price_pattern = r"\d+\.\d{2}(?=\s-\s\d+\s-\sPCS)"
        prices = []
        prices = re.findall(price_pattern, text)
        # st.write(prices)


        for hsn_code, quantity, description, edition_number, ske_code, price in zip(hsn_codes, qty_purchased, stripped_product_description, edition, ske_codes, prices):
            st.write(f"{ske_code} - {description} - {hsn_code} - {price} - {quantity}s")
        

        button_clicked = st.button('Approve')
        if button_clicked:
                if __name__ == '__main__':
                    loop = asyncio.ProactorEventLoop()
                    asyncio.set_event_loop(loop)
                    title=loop.run_until_complete(ske_go())
                    # print(title)



if option == 'BHAIRAV':

    st.header("Upload PDF 💬")
    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:

        st.write("The PDF Contains the following data")
        pdf_bytes = pdf.read()  # Read the raw content of the PDF file
        pdf_stream = BytesIO(pdf_bytes)  # Wrap it in a BytesIO object
        pdf_reader = PdfReader(pdf_stream)  # Pass the BytesIO object to PdfReader

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)
        # st.write(chunks)
        # st.write(type(text))
        text_string = ' '.join(text)
        # st.write(text)

        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectorstore = FAISS.from_texts(chunks, embeddings)

        chain = load_qa_chain(openai_instance, chain_type="stuff")



        # General Chat Bot Querry

        query_general = st.text_input("Please enter your prompt here.. ", "")
        docs_party = vectorstore.similarity_search(query_general)
        output = chain.run(input_documents=docs_party, question=query_general)
        st.write(output)
        




        # # Sample Working text - Only regex output for lines containing 'W' in vvipul bill - requires further cleaning

        # vipul_sample_pattern = r"\b\d+\sW.*"
        # vipul_sample_text_list = []
        # vipul_sample_text_list = re.findall(vipul_sample_pattern,text)
        # st.write(vipul_sample_text_list)
        # vipul_sample_text_string = ' '.join(vipul_sample_text_list)
        


        # #All Design Numbers in a Bill

        # vipul_design_number_pattern = r"D\d{4}"
        # all_vipul_design_number = []
        # all_vipul_design_number = re.findall(vipul_design_number_pattern, vipul_sample_text_string)
        # # st.write(all_vipul_design_number)


        # #Design Number Wise entries

        # # Dictionary to store split lists
        # split_lists = {design_number: [] for design_number in all_vipul_design_number}
        # split_lists["other"] = []  # For entries without the specified design numbers

        # # Split the list based on design numbers
        # for entry in vipul_sample_text_list:
        #     found_design_number = False
        #     for design_number in all_vipul_design_number:
        #         if design_number in entry:
        #             split_lists[design_number].append(entry)
        #             found_design_number = True
        #             break
        #     if not found_design_number:
        #         split_lists["other"].append(entry)

        # # Print the split lists
        # for key, value in split_lists.items():
        #     st.write(f"Entries with design number '{key}':")
        #     for entry in value:
        #         st.write(entry)
        #     st.write()
    



        # # Party Name
        # query_party_name = "From The Text - JUST mention the name of the seller"
        # docs_party = vectorstore.similarity_search(query_party_name)
        # party_name = chain.run(input_documents=docs_party, question=query_party_name)
        # st.write(f"Party Name: ",{party_name})

        
        



        # # Invoice Number
        # query_invoice_number = "From The Text - just mention the Invoice Number of the bill"
        # docs_party = vectorstore.similarity_search(query_invoice_number)
        # invoice_number = chain.run(input_documents=docs_party, question=query_invoice_number)
        # st.write(f"Invoice Number: {invoice_number}")
        # invoice_number_pattern = r"\b\d{4}\b"
        # invoice_number = re.findall(invoice_number_pattern, invoice_number)
        # # st.write(f"Last 4 digits - ",{invoice_number})


        # # All the Products in the bill
        # query_products = "LIST from the text in the following WAY - design number - item - hsn - rate - qty - UOM - value. GIVE IT TO ME AS A LIST"
        # docs_product_list = vectorstore.similarity_search(query_products)
        # # st.write(chain.run(input_documents=docs_product_list, question=query_products))
        # product_list = []
        # product_list = chain.run(input_documents=docs_product_list, question=query_products)
        # text = product_list
        # st.write(product_list)










# query = st.text_input("Ask questions about your PDF file:")
# docs = vectorstore.similarity_search(query)
# st.write(chain.run(input_documents=docs, question=query))

# # The Party/Sender - Name, GST IN, PAN Number

# query_party_name = "From The Text - JUST mention the name of the seller"
# docs_party = vectorstore.similarity_search(query_party_name)
# party_name = chain.run(input_documents=docs_party, question=query_party_name)
# st.write(f"Party Name: ",{party_name})

# query_party_gst = "From The Text - just mention the GSTIN of the seller"
# docs_party = vectorstore.similarity_search(query_party_gst)
# party_gst = chain.run(input_documents=docs_party, question=query_party_gst)
# st.write(f"Party GSTIN: ",{party_gst})


# # Pan not working
# query_party_pan = "From The Text - just mention the Companys PAN number - it is a ten character alfa-numeric text"
# docs_party = vectorstore.similarity_search(query_party_pan)
# party_pan = chain.run(input_documents=docs_party, question=query_party_pan)
# st.write(f"Party Pan: ",{party_pan})


# # Invoice Number
# query_invoice_number = "From The Text - just mention the Invoice Number of the bill"
# docs_party = vectorstore.similarity_search(query_invoice_number)
# invoice_number = chain.run(input_documents=docs_party, question=query_invoice_number)
# st.write(f"Invoice Number: ",{invoice_number})


# # The Invoice Date
# query_invoice_date = "From the text - list the invoice date"
# docs1 = vectorstore.similarity_search(query_invoice_date)
# invoice_date = chain.run(input_documents=docs1, question=query_invoice_date)
# st.write(f'Invoice date :', {invoice_date})



