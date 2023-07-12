/* encryptcontent/decrypt-contents.tpl.js */

/* Decrypts the content from the ciphertext bundle. */
function decrypt_content(password, iv_b64, ciphertext_b64, padding_char) {
    var key = CryptoJS.MD5(password),
        iv = CryptoJS.enc.Base64.parse(iv_b64),
        ciphertext = CryptoJS.enc.Base64.parse(ciphertext_b64),
        bundle = {
            key: key,
            iv: iv,
            ciphertext: ciphertext
        };
    var plaintext = CryptoJS.AES.decrypt(bundle, key, {
        iv: iv,
        padding: CryptoJS.pad.Pkcs7
    });
    try {
        return plaintext.toString(CryptoJS.enc.Utf8)
    } catch (err) {
        // encoding failed; wrong password
        return false;
    }
};

/* Split cyphertext bundle and try to decrypt it */
function decrypt_content_from_bundle(password, ciphertext_bundle) {
    // grab the ciphertext bundle and try to decrypt it
    if (ciphertext_bundle) {
        let parts = ciphertext_bundle.split(';');
        if (parts.length == 3) {
            return decrypt_content(password, parts[0], parts[1], parts[2]);
        }
    }
    return false;
};



/* Reload scripts src after decryption process */
function reload_js(src) {
    let script_src, script_tag, new_script_tag;
    let head = document.getElementsByTagName('head')[0];

    if (src.startsWith('#')) {
        script_tag = document.getElementById(src.substr(1));
        if (script_tag) {
            script_tag.remove();
            new_script_tag = document.createElement('script');
            if (script_tag.innerHTML) {
                new_script_tag.innerHTML = script_tag.innerHTML;
            }
            if (script_tag.src) {
                new_script_tag.src = script_tag.src;
            }
            head.appendChild(new_script_tag);
        }
    } else {
        if (base_url == '.') {
            script_src = src;
        } else {
            script_src = base_url + '/' + src;
        }

        script_tag = document.querySelector('script[src="' + script_src + '"]');
        if (script_tag) {
            script_tag.remove();
            new_script_tag = document.createElement('script');
            new_script_tag.src = script_src;
            head.appendChild(new_script_tag);
        }
    }
};



/* Decrypt speficique html entry from mkdocs configuration */
function decrypt_somethings(password_value, encrypted_something) {
    var html_item = '';
    for (const [name, tag] of Object.entries(encrypted_something)) {
        if (tag[1] == 'id') {
            html_item = [document.getElementById(name)];
        } else if (tag[1] == 'class') {
            html_item = document.getElementsByClassName(name);
        } else {
            console.log('WARNING: Unknow tag html found, check "encrypted_something" configuration.');
        }
        if (html_item[0]) {
            for (i = 0; i < html_item.length; i++) {
                // grab the cipher bundle if something exist
                let content = decrypt_content_from_bundle(password_value, html_item[i].innerHTML);
                if (content !== false) {
                    // success; display the decrypted content
                    html_item[i].innerHTML = content;
                    html_item[i].style.display = null;
                    // any post processing on the decrypted content should be done here
                }
            }
        }
    }
};

/* Decrypt content of a page */
function decrypt_action(password_input, encrypted_content, decrypted_content) {
    // grab the ciphertext bundle
    // and decrypt it
    let content = decrypt_content_from_bundle(password_input.value, encrypted_content.innerHTML);
    if (content !== false) {
        // success; display the decrypted content
        decrypted_content.innerHTML = content;
        // encrypted_content.parentNode.removeChild(encrypted_content);
        // any post processing on the decrypted content should be done here
        if (typeof MathJax === 'object') { MathJax.typesetPromise(); };
        
        
        
        return true
    } else {
        return false
    }
};

function decryptor_reaction(content_decrypted, password_input, fallback_used, set_global, save_cookie) {
    let location_path;
    if (set_global) {
        location_path = "/note/"; //global password decrypts at "/{site_path}"
    } else {
        location_path = encryptcontent_path;
    }
    if (content_decrypted) {
        
        // continue to decrypt others parts
        
        
    } else {
        // remove item on sessionStorage/localStorage if decryption process fail (Invalid item)
        if (!fallback_used || set_global) {
            // create HTML element for the inform message
            let mkdocs_decrypt_msg = document.getElementById('mkdocs-decrypt-msg');
            mkdocs_decrypt_msg.textContent = decryption_failure_message;
            
        }
        password_input.value = '';
        password_input.focus();
    }
}

/* Trigger decryption process */
function init_decryptor() {
    var password_input = document.getElementById('mkdocs-content-password');
    // adjust password field width to placeholder length
    if (password_input.hasAttribute('placeholder')) {
        password_input.setAttribute('size', password_input.getAttribute('placeholder').length);
    }
    var encrypted_content = document.getElementById('mkdocs-encrypted-content');
    var decrypted_content = document.getElementById('mkdocs-decrypted-content');
    
    
    /* Default, try decrypt content when key (ctrl) enter is press */
    password_input.addEventListener('keypress', function(event) {
        let set_global = false;
        if (event.key === "Enter") {
            if (event.ctrlKey) {
                set_global = true;
            }
            event.preventDefault();
            let content_decrypted = decrypt_action(
                password_input, encrypted_content, decrypted_content
            );
            decryptor_reaction(content_decrypted, password_input, false, set_global, true); //no fallback, set_global?, save cookie
        }
    });
}

document.addEventListener('DOMContentLoaded', init_decryptor());