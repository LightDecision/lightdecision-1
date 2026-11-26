import base64
import unittest
from lightdecision.api.assets import AssetResolver, IMAGE_MARKER, multipart_request
from lightdecision.api.typesafe import parse_request, format_answer

PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jL1kAAAAASUVORK5CYII=')
ASSET = {'format':'png','data_base64':base64.b64encode(PNG).decode()}

def body():
    return {'model':'lightdecision-1','state':{'ref-key':'ordinary','photo':'<image-x>'},
            'assets':{'<image-x>':ASSET},'questions':{
                'q':{'type':'choice','instructions':'Compare','criteria':{'<image-x>':'text only','other':'Example <image-x>'}},
                's':{'type':'score','instructions':'Rate <image-x>','criteria':['low',{'image':'<image-x>'}]}}}

def parse(b):return parse_request(b,model_ids={'lightdecision-1'})

class AssetsTests(unittest.TestCase):
    def test_position_scope_keys_and_score_legend(self):
        q,s=parse(body())
        self.assertIn(IMAGE_MARKER,q.decision.state)
        self.assertEqual(q.decision.options[0].name,'<image-x>')
        self.assertEqual(q.option_images[0],())
        self.assertIn(IMAGE_MARKER,q.decision.options[1].criteria)
        self.assertEqual(len(q.prefix_images),1)
        self.assertEqual(len(s.prefix_images),2)
        self.assertEqual(format_answer(s,[.1,.9])['legend']['1'],{'image':'<image-x>'})
        self.assertNotIn(ASSET['data_base64'],q.decision.state)

    def test_nested_order_repeated_references_and_whole_value_precedence(self):
        a=AssetResolver({'cat':ASSET,'category':ASSET})
        text,images=a.render({'z':'cat category cat','a':['category']})
        self.assertEqual(len(images),4);self.assertEqual(text.count(IMAGE_MARKER),4)
        a.finish()
        a=AssetResolver({'cat':ASSET});text,images=a.render({'cat':'ordinary'})
        self.assertEqual(images,());self.assertIn('cat',text)
        with self.assertRaises(ValueError):a.finish()

    def test_unused_invalid_assets_and_noncanonical_base64(self):
        b=body()
        for assets in [[],{' ':ASSET},{'unused':ASSET},{'<image-x>':{**ASSET,'type':'audio'}},
                       {'<image-x>':{**ASSET,'data_base64':'data:image/png;base64,'+ASSET['data_base64']}},
                       {'<image-x>':{**ASSET,'data_base64':ASSET['data_base64']+'\n'}},
                       {'<image-x>':{**ASSET,'format':'jpg'}}, {'<image-x>':{'part':'p'}}]:
            with self.subTest(assets=assets),self.assertRaises(ValueError):parse({**b,'assets':assets})

    def test_multipart_precedence_and_unused_parts(self):
        import json
        b=body();b['assets']['<image-x>']={**ASSET,'part':'picture','data_base64':'ignored'}
        data=(b'--boundary\r\nContent-Disposition: form-data; name="request"\r\nContent-Type: application/json\r\n\r\n'+json.dumps(b).encode()+
              b'\r\n--boundary\r\nContent-Disposition: form-data; name="picture"; filename="x.png"\r\nContent-Type: image/png\r\n\r\n'+PNG+b'\r\n--boundary--\r\n')
        result=multipart_request(data,'multipart/form-data; boundary=boundary')
        self.assertEqual(result['assets']['<image-x>']['data_base64'],ASSET['data_base64']);parse(result)
        for bad in [data.replace(b'name="request"',b'name="wrong"'),data.replace(b'name="picture"',b'name="other"'),data.replace(b'image/png',b'text/plain')]:
            with self.assertRaises(ValueError):multipart_request(bad,'multipart/form-data; boundary=boundary')

    def test_legacy_and_official_inputs_coexist_without_option_cap(self):
        b=body();b['images']=['/local/image.webp'];b['questions']['q']['criteria'].update({f'action{i}':None for i in range(1024)})
        q=parse(b)[0];self.assertEqual(len(q.decision.options),1026)
        self.assertEqual(len(format_answer(q,[1/1026]*1026)['probabilities']),1026)

if __name__=='__main__':unittest.main()
