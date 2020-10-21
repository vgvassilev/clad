---
title: "Project Meetings"
layout: gridlay
excerpt: "Project Meetings"
sitemap: false
permalink: /meetings/
---


# Project Meetings
{% for meeting in site.data.meetings %}

 <div class="well">
  <pubtit>{{ meeting.date }} at {{meeting.time_cest}} CEST</pubtit>
  Connection information: {{meeting.connect}} <br />
  Agenda:
  {% for item in meeting.agenda %}
   * {{item.title}} (<em>{{item.speaker}} </em>): [{{ item.link.display }}]({{ item.link.url }}) 
  {% endfor %}
 </div>
 
{% endfor %}
