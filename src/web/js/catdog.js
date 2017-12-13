/* implementation heavily influenced by http://bl.ocks.org/1166403 */

// define globals
var env = 'dev';
var i = 0;
var ddd = [];
var response = '';
var json_message = '';
var url = '';
var url_string = '';
var idname = 'chart_div';
var mapPolyLine = '';
var url_random_image = '';
//var icon_dict = {};
//var icon_dict['flag'] = 'https://developers.google.com/maps/documentation/javascript/examples/full/images/beachflag.png';
//var icon_dict['bus'] = 'http://maps.google.com/mapfiles/ms/icons/bus.png';
//var icon_dict['caution'] = 'http://maps.google.com/mapfiles/ms/icons/caution.png';
//var icon_dict['large_red'] = 'http://maps.google.com/mapfiles/ms/icons/large_red.png';

//var iconBase = 'https://maps.google.com/mapfiles/kml/shapes/';
//var icons = {
//  parking: {
//    icon: iconBase + 'parking_lot_maps.png'
//  },
//  library: {
//    icon: iconBase + 'library_maps.png'
//  },
//  info: {
//    icon: iconBase + 'info-i_maps.png'
//  }
//};
//




function syntaxHighlight(json) {
    if (typeof json != 'string') {
         json = JSON.stringify(json, undefined, 2);
    }
    json = json.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    return json.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, function (match) {
        var cls = 'number';
        if (/^"/.test(match)) {
            if (/:$/.test(match)) {
                cls = 'key';
            } else {
                cls = 'string';
            }
        } else if (/true|false/.test(match)) {
            cls = 'boolean';
        } else if (/null/.test(match)) {
            cls = 'null';
        }
        return '<span class="' + cls + '">' + match + '</span>';
    });
}


//url = 'http://35.161.103.202:5000';
url = 'http://www.dogorcat.online:5000';

function ReadTextEntry() {
//    url = document.getElementById("url").value;
    url_string = document.getElementById("url_string").value;
}


// Call the function
//var d = new Date();
//var now_time_us = d.getTime() * 1000;
//var time_window_size = 60e6;
//
//var d = new Date();
//var n = d.getTime();


var result = '';
var msg_parsed = {};
var json_parsed = '';

//var myVar = setInterval(deleteMarkers, 5000);
//
//function ClearJobId(callback, args){
//    document.getElementById("job_id_to_retrieve").value = '';
//    console.log('ClearJobId 1');
//    if (typeof callback == "function"){
//        if (typeof args == 'undefined'){
//            console.log('ClearJobId: callback, no args');
//            callback();
//        } else{
//            console.log('ClearJobId: callback, args');
//            callback(args);
//        }
//    }
//}

function myFunction(image_url) {
//    var x = document.createElement("IMG");
    x = document.getElementById("img");
    x.setAttribute("src", image_url);
    x.setAttribute("width", "500");
//    x.setAttribute("height", "228");
    x.setAttribute("alt", "image of cat or dog");
    document.body.appendChild(x);
}

function parseReponse(input){
    prob = input.prob;
    
    if (prob < 0.5){
        msg = 'I think this is a cat.\r\n(p = ' + prob + ')';
    }
    else {
        msg = 'I think this is a dog.\r\n(p = ' + prob + ')';
    }
    return(msg)
}


// get the data from the lambda function
function Random(){
    ReadTextEntry()
//    document.getElementById("demo").innerHTML = url + "<br>" + json_message;
    url_random_image = url + '/'  + 'random_image'
    jQuery.ajax( {
        type: 'GET',
        url: url + '/'  + 'random_image',
//        data: url_string,
        data: JSON.stringify({"url":url_string}),
        dataType: 'json',
//        contentType: "application/json; charset=utf-8",
        success: function( response ) {
            // response
            result = JSON.stringify(response);
            json_parsed = JSON.parse(result);
            
//            document.getElementById("invoke_response").innerHTML = syntaxHighlight(response);
            document.getElementById("url_string").value = response.url_n;
            Invoke();
        },
            
            error: function(xhr, ajaxOptions, thrownError) {
//                result = 'asdf';
            //error handling stuff
        }
    } );   
}



// get the data from the lambda function
function Invoke(){
    ReadTextEntry();
//    document.getElementById("demo").innerHTML = url + "<br>" + json_message;
    jQuery.ajax( {
        type: 'POST',
        url: url,
//        data: url_string,
        data: JSON.stringify({"url":url_string}),
        dataType: 'json',
//        contentType: "application/json; charset=utf-8",
        success: function( response ) {
            // response
    //        alert(JSON.stringify(response))
            result = JSON.stringify(response);
            json_parsed = JSON.parse(result);
            
//            document.getElementById("invoke_response").innerHTML = syntaxHighlight(response);
            document.getElementById("invoke_response").innerHTML = parseReponse(response);

            // show image of the cat or dog
            myFunction(json_parsed.url)
        },
            error: function(xhr, ajaxOptions, thrownError) {
//                result = 'asdf';
            //error handling stuff
        }
    } );   
}


