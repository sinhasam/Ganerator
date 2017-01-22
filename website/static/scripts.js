$(document).ready(function () {
	$(function () {
		var position = 0, scrollNav;
		$(window).scroll(function () {
			if($('.nav-slide').is(':animated')) return;
			scrollNav = $(window).scrollTop();
			if (scrollNav > position + 20 && scrollNav < position + 100) {
				 $('.nav-slide').slideUp(400);
			} else if (scrollNav < position) {
				 $('.nav-slide').slideDown(200);
			}
			position = scrollNav;
		});
	});
	$(function() {
	  $('#generate-button').bind('click', function() {
	    $.getJSON('http://127.0.0.1:5000/generate', {
	      }, function(data) {
	      $("#result").text(data.result);
	    });
	    return false;
	  });
	});
	$(function() {
	  $('#upload-button').bind('click', function() {
	    $.getJSON('http://127.0.0.1:5000/upload', {
	      }, function(data) {
	      $("#result").text(data.result);
	    });
	    return false;
	  });
	});
});
