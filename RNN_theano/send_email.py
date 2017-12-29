#-- include('examples/showgrabfullscreen.py') --#
import pyscreenshot as ImageGrab
import smtplib
from os.path import basename
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate
import time
from datetime import datetime
from datetime import timedelta

def send_mail(send_from, send_to, subject, text, files=None,
              server="smtp.gmail.com"):
    assert isinstance(send_to, list)

    msg = MIMEMultipart()
    msg['From'] = send_from
    msg['To'] = COMMASPACE.join(send_to)
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = subject

    msg.attach(MIMEText(text))

    for f in files or []:
        with open(f, "rb") as fil:
            part = MIMEApplication(
                fil.read(),
                Name=basename(f)
            )
            part['Content-Disposition'] = 'attachment; filename="%s"' % basename(f)
            msg.attach(part)


    smtp = smtplib.SMTP(server, 587)
    smtp.starttls()
    smtp.login("registerthisfakeguy@gmail.com", "13311331")
    smtp.sendmail(send_from, send_to, msg.as_string())
    smtp.close()

def schedule(h):
  now = datetime.now()
  run_at = now + timedelta(hours=h)
  delay = (run_at - now).total_seconds()
  while (delay > 0.0):
    if (delay % 60 == 0 and delay > 60):
      print "%d mins left to inform my lovely Master <3" % (delay/60)
    elif (delay < 60):
      print "%d seconds left to inform my lovely Master <3" % (delay)
    delay -= 1
    time.sleep(1)

  ImageGrab.grab_to_file("SS.png", childprocess=True, backend=None)
  send_mail("turing@localhost.com", ["tknbrk@gmail.com"],
              "The Screenshot You Desired, Master!",
              "Terminal made a progress.", ["SS.png"])
  print "Email has been sent to Master <3"

if __name__ == "__main__":
  #schedule(0.01)
  for i in range(8):
    schedule(0.5)
#-#
