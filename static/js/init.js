(function($){
  $(function(){

    var imageInput = $('#image');
    imageInput.hide();
    $('.button-collapse').sideNav();

    $('#start-button').click(function() {
      imageInput.click();
    });

    imageInput.change(function() {
      var img = document.getElementById('image')
      var formData = new FormData();
      formData.append("file", img.files[0])
      var request = new XMLHttpRequest();
      request.open('POST', '/predict');
      request.responseType = 'json';
      request.onload = function(e) {
        if (request.response.succ) {
          $('.col.s12.m3').css("background-color", "#ffffff");
          var clz = request.response['clazz'];
          $($('.col.s12.m3')[clz]).css("background-color", "#dadada");
        }
      };
      request.send(formData);

    });

  }); // end of document ready
})(jQuery); // end of jQuery name space
