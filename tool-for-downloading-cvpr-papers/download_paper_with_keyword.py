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


def get_cvpr_files_by_keywords(save_path, url, keywords):
    html = getHtml(url)
    parttern = re.compile(r'\bcontent.CVPR2021.*paper\.pdf\b')
    url_list = parttern.findall(html)

    filtered_url_list = []
    for url in url_list:
        name = url.split('/')[-1]
        is_download = False
        for keyword in keywords:
            if keyword in name:
                is_download = True
        if is_download:
            filtered_url_list.append(url)
    print('userful urls num:', len(filtered_url_list))

    for url in filtered_url_list:
        name = url.split('/')[-1]
        file_name = save_path + name
        download_file('http://openaccess.thecvf.com/' + url, file_name)


if __name__ == '__main__':
    # save_path = 'D:/paper_research/eccv2018/'
    # url = 'http://openaccess.thecvf.com/ECCV2018'
    # get_eccv_files(save_path, url)

    save_path = 'D:/paper_research/cvpr2021_gan/'
    url = 'https://openaccess.thecvf.com/CVPR2021?day=all'
    keywords = ['GAN', 'Generative Adversarial']
    get_cvpr_files_by_keywords(save_path, url, keywords)