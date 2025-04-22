import os
from OpenSSL import crypto

def create_self_signed_cert():
    # Create a key pair
    k = crypto.PKey()
    k.generate_key(crypto.TYPE_RSA, 4096)

    # Create a self-signed cert
    cert = crypto.X509()
    cert.get_subject().CN = "localhost"
    cert.set_serial_number(1000)
    cert.gmtime_adj_notBefore(0)
    cert.gmtime_adj_notAfter(365*24*60*60)  # 1 year validity
    cert.set_issuer(cert.get_subject())
    cert.set_pubkey(k)
    cert.sign(k, 'sha256')

    # Save certificate
    if not os.path.exists('ssl_cert'):
        os.makedirs('ssl_cert')
    
    with open("ssl_cert/cert.pem", "wb") as cert_file:
        cert_file.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
    
    with open("ssl_cert/key.pem", "wb") as key_file:
        key_file.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k))
    
    print("Certificate created successfully in ssl_cert/")

if __name__ == "__main__":
    create_self_signed_cert() 