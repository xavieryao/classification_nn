#!/usr/bin/python3
# -*- coding: utf-8 -*-
import requests
from progress.bar import Bar


def main(args):
    # get image list
    images = []

    print(">>> Getting image list......")
    estm = 200
    while True:
        url = 'https://api.cognitive.microsoft.com/bing/v5.0/images/search'
        headers = {
            'User-Agent': 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_1) AppleWebKit/602.2.14 (KHTML, like Gecko) Version/10.0.1 Safari/602.2.14',
            'Ocp-Apim-Subscription-Key': '203ffcebb20641ffa00dc021630a9b3b'
        }
        params = {
            'q': args.query_string,
            'offset': len(images),
            'mkt': 'zh-CN'
        }
        r = requests.get(url, headers=headers, params=params)
        res = r.json()
        if len(res['value']) == 0:
            break
        for v in res['value']:
            images.append({
                'url': v['contentUrl'],
                'id': v['imageId'],
                'encodingFormat': v['encodingFormat']
            })
        print('>>> {} added.'.format(len(res['value'])))
        if len(images) > estm:
            break

    print(">>> Got {} image urls!".format(len(images)))

    print(">>> Start downloading images...")
    bar = Bar("downloading images...", max=len(images),
              suffix="%(percent).1f%% - %(eta)ds")

    for idx, img in enumerate(images):
        r = requests.get(img['url'], stream=True)
        if r.status_code != requests.codes.ok:
            bar.next()
        else:
            with open("{}/{}.{}".format(args.folder, idx, img['encodingFormat']), 'wb') as fd:
                for chunk in r.iter_content(chunk_size=128):
                    fd.write(chunk)
            bar.next()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', "--query-string")
    parser.add_argument('-f', '--folder')
    main(parser.parse_args())
