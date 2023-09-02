import urllib.request
import re

def getHtml(url):
    page = urllib.request.urlopen(url)
    html = page.read()
    html = html.decode('utf-8')
    return html

def download_file(download_url, file_name):
    response = urllib.request.urlopen(download_url)
    file = open(file_name, 'wb')
    file.write(response.read())
    file.close()
    print("Completed")


def get_eccv_files(save_path, url):
    html = getHtml(url)
    parttern = re.compile(r'\bcontent_ECCV_2018.*paper\.pdf\b')
    url_list = parttern.findall(html)

    for url in url_list:
        name = url.split('/')[-1]
        file_name = save_path + name
        download_file('http://openaccess.thecvf.com/' + url, file_name)


def get_cvpr_files(save_path, url):
    html = getHtml(url)
    parttern = re.compile(r'\bcontent.CVPR2021.*paper\.pdf\b')
    url_list = parttern.findall(html)

    for url in url_list:
        name = url.split('/')[-1]
        file_name = save_path + name
        download_file('http://openaccess.thecvf.com/' + url, file_name)


if __name__ == '__main__':
    # save_path = 'D:/paper_research/eccv2018/'
    # url = 'http://openaccess.thecvf.com/ECCV2018'
    # get_eccv_files(save_path, url)

    save_path = 'D:/paper_research/cvpr2021/'
    url = 'https://openaccess.thecvf.com/CVPR2021?day=all'
    get_cvpr_files(save_path, url)

